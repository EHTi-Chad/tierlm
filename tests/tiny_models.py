"""Correctness test for every supported architecture, using no real model and almost no memory.

For each architecture it builds a tiny model with random weights, writes it as a GGUF, computes the logits
with a plain NumPy reference (written from the architecture's definition, independently of tierlm), runs
tierlm-logits on the same file and compares the two.

usage: python tests/tiny_models.py build/tierlm-logits [arch ...]
"""
import os, subprocess, sys, tempfile
import numpy as np
import gguf

V, CTX, BASE, EPS = 100, 512, 10000.0, 1e-5

# name: (gguf arch, settings)
CASES = {
    "llama":     ("llama",    dict(E=48, H=4, KV=2, HD=12, FF=40)),                          # GQA, adjacent-pair RoPE
    "llama3":    ("llama",    dict(E=48, H=4, KV=2, HD=12, FF=40, freqs=True)),              # + rope_freqs factors
    "mixtral":   ("llama",    dict(E=48, H=4, KV=4, HD=12, FFX=24, NE=8, NU=2, renorm=True)),
    "qwen2":     ("qwen2",    dict(E=48, H=4, KV=2, HD=12, FF=40, bias=True, tied=True)),
    "qwen3":     ("qwen3",    dict(E=48, H=4, KV=2, HD=16, FF=40, qkn="head")),              # head_dim != E/H
    "qwen3moe":  ("qwen3moe", dict(E=48, H=4, KV=2, HD=16, FFX=24, NE=8, NU=2, qkn="head", renorm=True)),
    "olmoe":     ("olmoe",    dict(E=48, H=4, KV=4, HD=12, FFX=24, NE=8, NU=2, qkn="full")),
}

def make(arch, s, seed, parts=1):
    rng = np.random.default_rng(seed)
    E, H, KV, HD = s["E"], s["H"], s["KV"], s["HD"]
    r = lambda *sh: (rng.standard_normal(sh) * 0.2).astype(np.float32)
    nw = lambda n: (1.0 + rng.standard_normal(n) * 0.1).astype(np.float32)
    L = 2
    W = {"token_embd.weight": r(V, E), "output_norm.weight": nw(E)}
    if not s.get("tied"): W["output.weight"] = r(V, E)
    if s.get("freqs"): W["rope_freqs.weight"] = (1.0 + rng.random(HD // 2) * 3).astype(np.float32)
    for i in range(L):
        b = f"blk.{i}."
        W.update({b + "attn_norm.weight": nw(E), b + "attn_q.weight": r(H * HD, E), b + "attn_k.weight": r(KV * HD, E),
                  b + "attn_v.weight": r(KV * HD, E), b + "attn_output.weight": r(E, H * HD), b + "ffn_norm.weight": nw(E)})
        if s.get("bias"):
            W.update({b + "attn_q.bias": r(H * HD), b + "attn_k.bias": r(KV * HD), b + "attn_v.bias": r(KV * HD)})
        if s.get("qkn") == "full": W.update({b + "attn_q_norm.weight": nw(H * HD), b + "attn_k_norm.weight": nw(KV * HD)})
        if s.get("qkn") == "head": W.update({b + "attn_q_norm.weight": nw(HD), b + "attn_k_norm.weight": nw(HD)})
        if "NE" in s:
            NE, F = s["NE"], s["FFX"]
            W.update({b + "ffn_gate_inp.weight": r(NE, E), b + "ffn_gate_exps.weight": r(NE, F, E),
                      b + "ffn_up_exps.weight": r(NE, F, E), b + "ffn_down_exps.weight": r(NE, E, F)})
        else:
            F = s["FF"]
            W.update({b + "ffn_gate.weight": r(F, E), b + "ffn_up.weight": r(F, E), b + "ffn_down.weight": r(E, F)})

    path = os.path.join(tempfile.mkdtemp(), f"tiny-{arch}.gguf")
    # parts > 1 writes a split model (tiny-x-00001-of-0000N.gguf ...), like big models on Hugging Face
    w = gguf.GGUFWriter(path, arch, split_max_tensors=-(-len(W) // parts) if parts > 1 else 0)
    w.add_block_count(L); w.add_embedding_length(E); w.add_head_count(H); w.add_head_count_kv(KV)
    w.add_key_length(HD); w.add_value_length(HD); w.add_rope_dimension_count(HD)
    w.add_context_length(CTX); w.add_rope_freq_base(BASE); w.add_layer_norm_rms_eps(EPS)
    if "NE" in s:
        w.add_expert_count(s["NE"]); w.add_expert_used_count(s["NU"])
        if arch == "qwen3moe": w.add_expert_feed_forward_length(s["FFX"]); w.add_feed_forward_length(s["FFX"] * 4)
        else: w.add_feed_forward_length(s["FFX"])
    else:
        w.add_feed_forward_length(s["FF"])
    for name, t in W.items(): w.add_tensor(name, t)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    if parts > 1:
        path = path[:-5] + f"-00001-of-{parts:05d}.gguf"
        assert os.path.exists(path), "split writer used an unexpected file name"
    return path, W, L

def rms(x, g): return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * g
def softmax(x): x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)

def rope(x, pos, neox, factors):       # x: [T, heads, HD]
    HD = x.shape[-1]
    inv = BASE ** (-np.arange(0, HD, 2) / HD)
    if factors is not None: inv = inv / factors
    ang = pos[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    if neox:                            # rotate halves
        a, b = x[..., :HD // 2], x[..., HD // 2:]
        return np.concatenate([a * c - b * s, b * c + a * s], -1)
    a, b = x[..., 0::2], x[..., 1::2]   # rotate adjacent pairs (original Llama layout in GGUF)
    out = np.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = b * c + a * s
    return out

def reference(arch, s, W, L, ids):
    E, H, KV, HD = s["E"], s["H"], s["KV"], s["HD"]
    T = len(ids); pos = np.arange(T, dtype=np.float64)
    neox = arch != "llama"
    factors = W.get("rope_freqs.weight")
    x = W["token_embd.weight"][ids].astype(np.float64)
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(L):
        b = f"blk.{i}."
        h = rms(x, W[b + "attn_norm.weight"])
        q, k, v = h @ W[b + "attn_q.weight"].T, h @ W[b + "attn_k.weight"].T, h @ W[b + "attn_v.weight"].T
        if s.get("bias"): q, k, v = q + W[b + "attn_q.bias"], k + W[b + "attn_k.bias"], v + W[b + "attn_v.bias"]
        if s.get("qkn") == "full": q, k = rms(q, W[b + "attn_q_norm.weight"]), rms(k, W[b + "attn_k_norm.weight"])
        q, k, v = q.reshape(T, H, HD), k.reshape(T, KV, HD), v.reshape(T, KV, HD)
        if s.get("qkn") == "head": q, k = rms(q, W[b + "attn_q_norm.weight"]), rms(k, W[b + "attn_k_norm.weight"])
        q, k = rope(q, pos, neox, factors), rope(k, pos, neox, factors)
        k, v = np.repeat(k, H // KV, axis=1), np.repeat(v, H // KV, axis=1)   # grouped-query attention
        att = softmax(np.einsum("qhd,khd->hqk", q, k) / np.sqrt(HD) + mask)
        o = np.einsum("hqk,khd->qhd", att, v).reshape(T, H * HD)
        x = x + o @ W[b + "attn_output.weight"].T
        h = rms(x, W[b + "ffn_norm.weight"])
        silu = lambda z: z / (1 + np.exp(-z))
        if "NE" not in s:
            x = x + (silu(h @ W[b + "ffn_gate.weight"].T) * (h @ W[b + "ffn_up.weight"].T)) @ W[b + "ffn_down.weight"].T
            continue
        p = softmax(h @ W[b + "ffn_gate_inp.weight"].T)
        out = np.zeros_like(x)
        for t in range(T):
            top = np.argsort(-p[t], kind="stable")[:s["NU"]]
            wts = p[t, top] / (p[t, top].sum() if s.get("renorm") else 1.0)
            for e, wt in zip(top, wts):
                g = W[b + "ffn_gate_exps.weight"][e] @ h[t]
                u = W[b + "ffn_up_exps.weight"][e] @ h[t]
                out[t] += wt * (W[b + "ffn_down_exps.weight"][e] @ (silu(g) * u))
        x = x + out
    x = rms(x, W["output_norm.weight"])
    return x[-1] @ W.get("output.weight", W["token_embd.weight"]).T

# Two passes: exact (f32 KV cache, must match to ~1e-6) and the normal f16 cache (rounding only, same top token).
names = sys.argv[2:] or list(CASES)
# With TIERLM_RPC set (a tierlm-rpc-server standing in for a GPU), every case also runs split between the
# device and RAM at several VRAM budgets (fractions of the model size), and must give the same results.
SPLITS = [(mode, f) for mode in ("layers", "experts") for f in (0.0, 0.15, 0.4, 0.7, 2.0)] if os.environ.get("TIERLM_RPC") else [None]

# Two passes:
#   exact   on the CPU with an f32 KV cache: the math itself, which must match the reference to ~1e-5.
#           (With a stand-in RPC GPU for split tests, on that device: it computes exactly too.)
#   normal  the usual settings on the default device (a GPU if there is one): f16 KV cache, and GPU kernels that
#           round (flash attention in f16, TF32 tensor cores for small batches on RTX 30-series and newer), so
#           differences up to a few % of the logit range are expected; the top token must still match. Real bugs
#           show up as errors of 100% or more.
def PASSES(extra):
    exact_env = {"TIERLM_KV_F32": "1", **extra}
    if not os.environ.get("TIERLM_RPC") and "TIERLM_DEVICE" not in os.environ:
        exact_env["TIERLM_DEVICE"] = "cpu"
    return (("exact", exact_env, 1e-4), ("normal", dict(extra), 6e-2))
long_ids = list(np.random.default_rng(7).integers(0, V, 300))   # > 256: exercises batched prompts
ok = True
for idx, name in enumerate(names):
    arch, s = CASES[name]
    path, W, L = make(arch, s, 100 + list(CASES).index(name))   # same random model however the tests are picked
    size = os.path.getsize(path)
    for frac in SPLITS:
      split_env = {} if frac is None else {"TIERLM_SPLIT": frac[0], "TIERLM_VRAM_BYTES": str(int(size * frac[1]))}
      for label, env, tol in PASSES(split_env):
        worst, all_top = 0.0, True
        placement = ""
        for mode in ([], ["--step"]):
            for ids in ([5], [1, 2, 3], [7, 42, 99, 0, 13, 13, 64, 8, 21, 55], long_ids):
                want = reference(arch, s, W, L, np.array(ids))
                run = subprocess.run([sys.argv[1], path, *mode, *map(str, ids)], capture_output=True, text=True,
                                     env={**os.environ, **env, "TIERLM_SHOW_PLACEMENT": "1"})
                if run.returncode != 0:
                    sys.exit(f"{name}: tierlm-logits failed: {run.stderr.strip()}")
                placement = next((l for l in run.stderr.splitlines() if "VRAM" in l), "")
                got = np.array([float(v) for v in run.stdout.split()])
                worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
                all_top &= bool(got.argmax() == want.argmax())
        passed = worst < tol and all_top
        ok &= passed
        where = "" if frac is None else f" [{placement.split(': ', 1)[-1]}]"
        print(f"{name:9s} {label:9s}: worst relative diff {worst:.1e} (limit {tol:.0e}), top token matches: {all_top}  ->  {'ok' if passed else 'FAIL'}{where}")

# Split models: the same weights written as 3 files, loaded through part 1, must give the same results.
if not os.environ.get("TIERLM_RPC"):
    for name in names:
        arch, s = CASES[name]
        path, W, L = make(arch, s, 100 + list(CASES).index(name), parts=3)
        worst, all_top = 0.0, True
        for ids in ([5], [7, 42, 99, 0, 13, 13, 64, 8, 21, 55]):
            want = reference(arch, s, W, L, np.array(ids))
            run = subprocess.run([sys.argv[1], path, *map(str, ids)], capture_output=True, text=True,
                                 env={**os.environ, "TIERLM_KV_F32": "1", **({} if "TIERLM_DEVICE" in os.environ else {"TIERLM_DEVICE": "cpu"})})
            if run.returncode != 0:
                sys.exit(f"{name} (3 files): tierlm-logits failed: {run.stderr.strip()}")
            got = np.array([float(v) for v in run.stdout.split()])
            worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
            all_top &= bool(got.argmax() == want.argmax())
        passed = worst < 1e-4 and all_top
        ok &= passed
        print(f"{name:9s} 3 files  : worst relative diff {worst:.1e} (limit 1e-04), top token matches: {all_top}  ->  {'ok' if passed else 'FAIL'}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
