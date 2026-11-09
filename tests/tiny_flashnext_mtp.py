"""Qwen3.8 Flash-Next's MTP draft block (multi-token prediction) against a reference built from Hugging Face's own
layers.

Hugging Face's Qwen4Exp code skips the draft block's weights (mtp.*), so the reference is assembled here from its
pieces, as the block is laid out in the released files: at position t the model's hyper-connection copies before its
last mixer (h_{t-1}; zeros at t = 0) are normed copy by copy (nextn.hnorm), each one beside the normed embedding of
the token at t (nextn.enorm) goes through nextn.eh_proj, which gives the block's own copies; then one full-attention
decoder layer of the model's kind (Hugging Face's Qwen4ExpTextDecoderLayer, its stream mixers included), then the
block's own last mixer (nextn.hc_head_*, Hugging Face's Qwen4ExpTextGatedResidual without the inject weights) and
the model's output head. It predicts the token at t + 1.

The GGUF holds the block as layer n_layer (blk.<n_layer>.*, nextn_predict_layers = 1), as Unsloth's files do.
tierlm-logits --mtp runs the block over every position at once, --mtp-step one position at a time.

usage: python tests/tiny_flashnext_mtp.py build/tierlm-logits
"""
import os, subprocess, sys, tempfile, types
import numpy as np
import torch
import gguf
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (Qwen4ExpForCausalLM, Qwen4ExpTextDecoderLayer,
                                                              Qwen4ExpTextGatedResidual)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("tfn", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_flashnext.py"))
# (tiny_flashnext.py runs its tests when imported: take only what's needed from its source)
src = open(spec.origin).read().split("\ndef ref_logits")[0]
tfn = {}
exec(compile(src, spec.origin, "exec"), tfn)
BASE, V, EOS, write_base = tfn["BASE"], tfn["V"], tfn["EOS"], tfn["write"]

def build(over, seed):
    torch.manual_seed(seed)
    cfg = Qwen4ExpTextConfig(**{**BASE, **over})
    m = Qwen4ExpForCausalLM(cfg).eval()
    li = cfg.num_hidden_layers - 1                       # a full-attention layer's kind (the last one is)
    assert cfg.layer_types[li] != "linear_attention"
    blk = Qwen4ExpTextDecoderLayer(cfg, li).eval()
    head = Qwen4ExpTextGatedResidual(cfg, use_combine=False).eval()
    E, HC = cfg.hidden_size, cfg.hc_count
    eh = torch.nn.Linear(2 * E, E, bias=False)
    with torch.no_grad():
        for mod in (m, blk, head, eh):
            for n, p in mod.named_parameters():
                if n.endswith("A_log"): p.copy_(torch.log(torch.empty_like(p).uniform_(0.5, 4)))
                elif n.endswith("dt_bias"): p.copy_(0.5 + 0.1 * torch.randn_like(p))
                elif "linear_attn.norm" in n: p.copy_(1 + 0.1 * torch.randn_like(p))
                elif "norm" in n: p.copy_(0.1 * torch.randn_like(p))
                else: p.copy_(0.2 * torch.randn_like(p))
    mtp = dict(blk=blk, head=head, eh=eh,
               enorm=(1 + 0.1 * torch.randn(E)).numpy(), hnorm=(1 + 0.1 * torch.randn(HC * E)).numpy())
    return cfg, m, mtp

def write(cfg, m, mtp):
    # The model as tiny_flashnext.py writes it, with the block count one higher, the draft block's settings and its
    # tensors added (a GGUFWriter that adds them on the way).
    n, A, cr = cfg.num_hidden_layers, "qwen4exp", cfg.indexer_compress_ratio
    T = block_tensors(cfg, mtp)
    class W(gguf.GGUFWriter):
        def add_block_count(self, k): super().add_block_count(k + 1)
        def add_array(self, key, val):
            super().add_array(key, list(val) + [cr] if key.endswith("compress_ratios") else val)
        def write_header_to_file(self, *a, **kw):
            self.add_uint32(f"{A}.nextn_predict_layers", 1)
            for name, t in T.items(): self.add_tensor(name, np.ascontiguousarray(t).astype(np.float32))
            super().write_header_to_file(*a, **kw)
    mod = types.ModuleType("gguf_mtp"); mod.__dict__.update(gguf.__dict__); mod.GGUFWriter = W
    tfn["gguf"] = mod
    try: path, _ = write_base(cfg, m)
    finally: tfn["gguf"] = gguf
    return path

def split_companion(path, n):
    # The same model as two files, as Unsloth publishes it: the model without its draft block, and
    # "mtp-<name>.gguf" next to it holding only the draft block (its own block count, and no token table: the
    # "shared" kind reuses the model's).
    r = gguf.GGUFReader(path)
    d = tempfile.mkdtemp()
    main, comp = os.path.join(d, "tiny.gguf"), os.path.join(d, "mtp-tiny-shared.gguf")
    for out, keep, drop_nextn in ((main, lambda t: not t.startswith(f"blk.{n}."), True), (comp, lambda t: t.startswith(f"blk.{n}."), False)):
        w = gguf.GGUFWriter(out, "qwen4exp")
        for f in r.fields.values():
            if f.name.startswith("GGUF.") or f.name == "general.architecture": continue
            val = f.contents()
            if f.name == "qwen4exp.block_count" and drop_nextn: val = n
            if f.name == "qwen4exp.nextn_predict_layers" and drop_nextn: continue
            if f.name == "qwen4exp.attention.compress_ratios" and drop_nextn: val = val[:n]
            sub = f.types[-1] if f.types[0] == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(f.name, val, f.types[0], sub)
        for t in r.tensors:
            if keep(t.name): w.add_tensor(t.name, np.array(t.data).reshape(list(reversed([int(x) for x in t.shape]))), raw_dtype=t.tensor_type)
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return main, comp

def block_tensors(cfg, mtp):
    n = cfg.num_hidden_layers
    sd = {k: v.detach().float().numpy() for k, v in mtp["blk"].state_dict().items()}
    b = f"blk.{n}."
    T = {}
    for k, nm in (("attn", "attn_hyper_connection"), ("ffn", "mlp_hyper_connection")):
        T[b + f"hc_{k}_norm.weight"] = 1 + sd[nm + ".hc_norm.weight"]
        T[b + f"hc_{k}_down.weight"] = sd[nm + ".input_mix_weight_down.weight"]
        T[b + f"hc_{k}_up.weight"] = sd[nm + ".input_mix_weight_up.weight"]
        T[b + f"hc_{k}_inject.weight"] = sd[nm + ".block_inject_weight.weight"]
    a = "self_attn."
    T.update({b + "attn_q.weight": sd[a + "q_proj.weight"], b + "attn_k.weight": sd[a + "k_proj.weight"],
              b + "attn_v.weight": sd[a + "v_proj.weight"], b + "attn_output.weight": sd[a + "o_proj.weight"],
              b + "attn_q_norm.weight": 1 + sd[a + "q_norm.weight"], b + "attn_k_norm.weight": 1 + sd[a + "k_norm.weight"]})
    qk = sd[a + "indexer.index_qk_proj.weight"]; nq = cfg.indexer_n_heads * cfg.indexer_head_dim
    T.update({b + "indexer.q_proj.weight": qk[:nq], b + "indexer.k_proj.weight": qk[nq:],
              b + "indexer.q_norm.weight": 1 + sd[a + "indexer.q_layernorm.weight"], b + "indexer.k_norm.weight": 1 + sd[a + "indexer.k_layernorm.weight"]})
    mp = "mlp."
    gu = sd[mp + "experts.gate_up_proj"]; F = gu.shape[1] // 2
    T.update({b + "ffn_gate_inp.weight": sd[mp + "gate.weight"], b + "ffn_gate_exps.weight": np.ascontiguousarray(gu[:, :F, :]),
              b + "ffn_up_exps.weight": np.ascontiguousarray(gu[:, F:, :]), b + "ffn_down_exps.weight": sd[mp + "experts.down_proj"],
              b + "ffn_gate_shexp.weight": sd[mp + "shared_expert.gate_proj.weight"], b + "ffn_up_shexp.weight": sd[mp + "shared_expert.up_proj.weight"],
              b + "ffn_down_shexp.weight": sd[mp + "shared_expert.down_proj.weight"], b + "ffn_gate_inp_shexp.weight": sd[mp + "shared_expert_gate.weight"].reshape(-1)})
    hs = {k: v.detach().float().numpy() for k, v in mtp["head"].state_dict().items()}
    T.update({b + "nextn.eh_proj.weight": mtp["eh"].weight.detach().float().numpy(),
              b + "nextn.enorm.weight": mtp["enorm"], b + "nextn.hnorm.weight": mtp["hnorm"],
              b + "nextn.hc_head_norm.weight": 1 + hs["hc_norm.weight"],
              b + "nextn.hc_head_down.weight": hs["input_mix_weight_down.weight"],
              b + "nextn.hc_head_up.weight": hs["input_mix_weight_up.weight"]})
    return T

def mtp_reference(cfg, m, mtp, ids):
    E, HC = cfg.hidden_size, cfg.hc_count
    got = {}
    full = [i for i, t in enumerate(cfg.layer_types) if t != "linear_attention"][0]
    h1 = m.model.hyper_connection_mixer.register_forward_pre_hook(lambda mod, args: got.__setitem__("X", args[0]))
    h2 = m.model.layers[full].register_forward_pre_hook(lambda mod, args, kw: got.update(pe=kw["position_embeddings"], mask=kw["attention_mask"]), with_kwargs=True)
    with torch.no_grad():
        m(torch.tensor([ids]))
    h1.remove(); h2.remove()
    X = got["X"][0]                                       # [T, HC*E], before the last mixer
    T_ = len(ids)
    hprev = torch.cat([torch.zeros(1, HC * E), X[:-1]])
    def rms(x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    with torch.no_grad():
        e = m.model.embed_tokens(torch.tensor(ids)).float()
        en = rms(e) * torch.tensor(mtp["enorm"])                                       # [T, E]
        hn = (rms(hprev.reshape(T_, HC, E)).reshape(T_, HC * E) * torch.tensor(mtp["hnorm"])).reshape(T_, HC, E)
        x = mtp["eh"](torch.cat([en[:, None, :].expand(T_, HC, E), hn], -1)).reshape(1, T_, HC * E)
        x = mtp["blk"](x, position_embeddings=got["pe"], attention_mask=got["mask"], conv_mask=None, past_key_values=None)
        o = mtp["head"](x)                                                             # [1, T, E]
        return m.lm_head(o)[0, -1].double().numpy()

exe = sys.argv[1]
long_ids = [int(x) for x in np.random.default_rng(3).integers(2, V, 200)]
long_ids[37] = long_ids[120] = EOS
tests = [[5], [1, 2, 3], [7, 42, 99, 0, 13, 13, 64, 8, 21, 55, 117, EOS, 4, 4, 9], long_ids]
CASES = {"mtp": dict(), "mtp-8layer": dict(num_hidden_layers=8)}
ok = True
for ci, (name, over) in enumerate(CASES.items()):
    cfg, m, mtp = build(over, 50 + ci)
    path = write(cfg, m, mtp)
    for label, env, tol in (("exact", {"TIERLM_KV_F32": "1", "TIERLM_DEVICE": "cpu"}, 1e-4), ("normal", {}, 6e-2)):
        worst, top = 0.0, True
        for mode in (["--mtp"], ["--mtp-step"]):
            for ids in tests:
                want = mtp_reference(cfg, m, mtp, ids)
                r = subprocess.run([exe, path, *mode, *map(str, ids)], capture_output=True, text=True, env={**os.environ, **env})
                if r.returncode != 0: sys.exit(f"{name}: tierlm-logits {mode[0]} failed: {r.stderr.strip()[-800:]}")
                got = np.array([float(v) for v in r.stdout.split()])
                worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
                top &= bool(got.argmax() == want.argmax())
        passed = worst < tol and top
        ok &= passed
        print(f"{name:12s} {label:7s}: worst relative diff {worst:.1e} (limit {tol:.0e}), top token matches: {top}  ->  {'ok' if passed else 'FAIL'}")
        # Speculative decoding with the draft block: greedy output must not change, with its own guesses (poor on
        # random weights) and with scripted guesses wrong at different depths, so undo runs at every depth.
        def gen(mode, extra=None):
            r = subprocess.run([exe, path, mode, "60", "5", "9", "13"], capture_output=True, text=True, env={**os.environ, **env, **(extra or {})})
            if r.returncode != 0: sys.exit(f"{name}: tierlm-logits {mode} failed: {r.stderr.strip()[-800:]}")
            return r.stdout.split(), r.stderr.strip().splitlines()[-1:] if r.stderr.strip() else []
        plain, _ = gen("--gen")
        checks = [gen("--gen-spec")] + [gen("--gen-spec", {"TIERLM_DRAFT_SCRIPT": ",".join(plain), "TIERLM_DRAFT_CORRUPT": k}) for k in ("3", "5")]
        checks += [gen("--gen-spec", {"TIERLM_DRAFT_N": "8", "TIERLM_DRAFT_SCRIPT": ",".join(plain), "TIERLM_DRAFT_CORRUPT": k}) for k in ("2", "7")]
        same = all(c[0] == plain for c in checks)
        ok &= same
        print(f"{name:12s} {label:7s}: greedy output with speculation {'identical' if same else 'DIFFERENT'} ({'; '.join(' '.join(c[1]) for c in checks)})  ->  {'ok' if same else 'FAIL'}")
    # The draft block from a separate "mtp-" file next to the model: found, and the same predictions and output.
    main, comp = split_companion(path, cfg.num_hidden_layers)
    env = {"TIERLM_KV_F32": "1", "TIERLM_DEVICE": "cpu"}
    same = True
    for mode in (["--mtp", "1", "2", "3"], ["--gen-spec", "40", "5", "9", "13"]):
        a = subprocess.run([exe, path, *mode], capture_output=True, text=True, env={**os.environ, **env})
        b = subprocess.run([exe, main, *mode], capture_output=True, text=True, env={**os.environ, **env})
        if b.returncode != 0: sys.exit(f"{name}: tierlm-logits {mode[0]} with the separate draft file failed: {b.stderr.strip()[-800:]}")
        found = "draft block from" in b.stderr
        same &= found and a.stdout == b.stdout and a.returncode == 0
    ok &= same
    print(f"{name:12s} {'file':7s}: separate mtp- draft file {'found, same results' if same else 'NOT the same'}  ->  {'ok' if same else 'FAIL'}")
    os.remove(path)

# Speculative sampling (temperature > 0): guesses drawn from the draft block's probabilities and kept by the ratio of
# the model's to the draft block's must leave the replies' odds exactly as the model's own. On a 16-token vocabulary
# (so the guesses are often kept: about 40%), many 3-token replies are drawn with and without guessing and compared
# with a chi-square test, token by token, in pairs and whole (z within 4: a real error shows up far beyond that).
from collections import Counter
cfg, m, mtp = build(dict(vocab_size=16), 60)
path = write(cfg, m, mtp)
env = {**os.environ, "TIERLM_DEVICE": "cpu", "TIERLM_SAMPLE_RUNS": "8000", "TIERLM_SAMPLE_TOPK": "16", "TIERLM_SAMPLE_TEMP": "1"}
runs = {}
for mode in ("--gen", "--gen-spec"):
    r = subprocess.run([exe, path, mode, "3", "5", "9", "13"], capture_output=True, text=True, env=env)
    if r.returncode != 0: sys.exit(f"sampling: tierlm-logits {mode} failed: {r.stderr.strip()[-800:]}")
    runs[mode] = [tuple(l.split(",")) for l in r.stdout.split()]
def z_of(a, b):
    ca, cb = Counter(a), Counter(b)
    na, nb = len(a), len(b)
    chi, df, rest = 0.0, -1, [0, 0]
    for k in set(ca) | set(cb):
        x, y = ca[k], cb[k]
        if x + y < 10: rest[0] += x; rest[1] += y; continue
        ex, ey = (x + y) * na / (na + nb), (x + y) * nb / (na + nb)
        chi += (x - ex) ** 2 / ex + (y - ey) ** 2 / ey; df += 1
    if sum(rest):
        x, y = rest; ex, ey = (x + y) * na / (na + nb), (x + y) * nb / (na + nb)
        chi += (x - ex) ** 2 / ex + (y - ey) ** 2 / ey; df += 1
    return (chi - df) / (2 * df) ** 0.5 if df > 0 else 0.0
P, Q = runs["--gen"], runs["--gen-spec"]
zs = [z_of([t[i] for t in P], [t[i] for t in Q]) for i in range(3)]
zs += [z_of([t[:2] for t in P], [t[:2] for t in Q]), z_of([t[1:] for t in P], [t[1:] for t in Q]), z_of(P, Q)]
passed = all(abs(z) < 4 for z in zs)
ok &= passed
print(f"{'sampling':12s} {'exact':7s}: replies' odds with speculative sampling, z = {', '.join(f'{z:.1f}' for z in zs)} (limit 4)  ->  {'ok' if passed else 'FAIL'}")
os.remove(path)
sys.exit(0 if ok else 1)
