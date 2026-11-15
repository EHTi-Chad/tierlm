"""Correctness test for DeepSeek V4 (GGUF arch "deepseek4", e.g. DeepSeek-V4-Flash), with no real model.

What's new in V4, all of it in this tiny model:
  hyper-connections  the residual stream is 4 streams; each block reads a weighted mix of them (pre), writes its
                     output into each with its own weight (post), and mixes the streams (comb, a doubly stochastic
                     4x4 matrix from 20 Sinkhorn steps). The weights come from the streams themselves.
  attention          one key/value head shared by all query heads (keys are values), partial interleaved RoPE on
                     the last n_rot dims, a learned "sink" per head, the output rotated back by its own position,
                     and a grouped low-rank output projection. Each token sees the last `window` tokens as they
                     are, plus compressed entries of everything older:
     CSA layers      every 4 tokens -> one entry (a softmax-gated sum over 8 tokens: this window's and the one
                     before's), and a small "indexer" picks the top_k entries each token attends to
     HCA layers      every 128 tokens (here 16) -> one entry, all of them attended to
  experts            sqrt(softplus) scores, the first layers routed by a fixed token -> experts table (hash),
                     the rest by score + bias; SwiGLU with clamped inputs; one shared expert

The math follows Hugging Face's modeling_deepseek_v4.py (transformers 5.23) without its cache (the whole
sequence at once); tensor names and layouts are those of the GGUF files (llama.cpp's converter).

usage: python tests/tiny_deepseek4.py build/tierlm-logits
       python tests/tiny_deepseek4.py --write out.gguf
"""
import math, os, subprocess, sys, tempfile
import numpy as np
import gguf

rng = np.random.default_rng(404)
V, E, CTX, EPS, HCEPS = 96, 32, 512, 1e-6, 1e-6
HC, SINK_IT = 4, 20
H, HD, ROT, QL = 4, 32, 8, 16              # query heads, head dim (keys = values), rope dims (the last ones), q rank
OG, OL = 2, 8                             # output groups, rank per group
W = 8                                     # sliding window (the real models: 128)
IH, IHD, TOPK = 8, 16, 3                  # indexer heads, head dim, entries kept per token
NE, NU, FE, WSCALE = 8, 2, 16, 1.5        # experts, per token, width, weights scale
HASH = 1                                  # layers routed by the token table
RATIOS = [0, 4, 16, 4, 16, 0]             # per layer: 0 sliding only, 4 CSA, 16 HCA (the real models: 4 / 128)
CLAMP = [0.8, 6.0, 0.9, 6.0, 0.7, 6.0]    # SwiGLU clamp per layer (experts; the shared expert uses CLAMP_S)
CLAMP_S = [6.0, 0.8, 6.0, 0.9, 6.0, 0.7]
THETA, CTHETA = 10000.0, 160000.0         # RoPE base: sliding-only layers; compressed layers (with YaRN)
YF, YORIG, YBF, YBS = 16.0, 64, 32.0, 1.0  # YaRN: factor, original context, beta fast / slow
L = len(RATIOS)
r = lambda *s: (rng.standard_normal(s) * 0.2).astype(np.float32)
nw = lambda n: (1 + rng.standard_normal(n) * 0.1).astype(np.float32)

P = {"embed": r(V, E), "norm": nw(E), "head": r(V, E),
     "hh_fn": r(HC, HC * E), "hh_base": r(HC), "hh_scale": (1 + r(1)).astype(np.float32)}
for i, m in enumerate(RATIOS):
    p = {"attn_norm": nw(E), "ffn_norm": nw(E),
         "q_a": r(QL, E), "q_a_norm": nw(QL), "q_b": r(H * HD, QL), "kv": r(HD, E), "kv_norm": nw(HD),
         "o_a": r(OG * OL, H * HD // OG), "o_b": r(E, OG * OL), "sinks": r(H),
         "router": r(NE, E), "eg": r(NE, FE, E), "eu": r(NE, FE, E), "ed": r(NE, E, FE),
         "sg": r(FE, E), "su": r(FE, E), "sd": r(E, FE)}
    for site in ("attn", "ffn"):
        p[site + "_hc_fn"] = r((2 + HC) * HC, HC * E)
        p[site + "_hc_base"] = r((2 + HC) * HC)
        p[site + "_hc_scale"] = (1 + r(3)).astype(np.float32)
    if i < HASH: p["tid2eid"] = np.stack([rng.permutation(NE)[:NU] for _ in range(V)]).astype(np.int32)
    else: p["probs_b"] = r(NE) * 0.5
    if m:
        two = 2 if m == 4 else 1                  # CSA: two series per token (this window's, the next one's)
        p.update({"c_kv": r(two * HD, E), "c_gate": r(two * HD, E), "c_ape": r(m, two * HD), "c_norm": nw(HD)})
        if m == 4:
            p.update({"i_q_b": r(IH * IHD, QL), "i_proj": r(IH, E),
                      "ic_kv": r(2 * IHD, E), "ic_gate": r(2 * IHD, E), "ic_ape": r(4, 2 * IHD), "ic_norm": nw(IHD)})
    P[i] = p

def write_gguf(path=None):
    T = {"token_embd.weight": P["embed"], "output_norm.weight": P["norm"], "output.weight": P["head"],
         "output_hc_fn.weight": P["hh_fn"], "output_hc_base.weight": P["hh_base"], "output_hc_scale.weight": P["hh_scale"]}
    for i, m in enumerate(RATIOS):
        p, b = P[i], f"blk.{i}."
        T.update({b + "attn_norm.weight": p["attn_norm"], b + "ffn_norm.weight": p["ffn_norm"],
                  b + "attn_q_a.weight": p["q_a"], b + "attn_q_a_norm.weight": p["q_a_norm"], b + "attn_q_b.weight": p["q_b"],
                  b + "attn_kv.weight": p["kv"], b + "attn_kv_a_norm.weight": p["kv_norm"],
                  b + "attn_output_a.weight": p["o_a"], b + "attn_output_b.weight": p["o_b"], b + "attn_sinks.weight": p["sinks"],
                  b + "ffn_gate_inp.weight": p["router"], b + "ffn_gate_exps.weight": p["eg"], b + "ffn_up_exps.weight": p["eu"],
                  b + "ffn_down_exps.weight": p["ed"], b + "ffn_gate_shexp.weight": p["sg"], b + "ffn_up_shexp.weight": p["su"],
                  b + "ffn_down_shexp.weight": p["sd"]})
        for site in ("attn", "ffn"):
            T[b + f"hc_{site}_fn.weight"] = p[site + "_hc_fn"]
            T[b + f"hc_{site}_base.weight"] = p[site + "_hc_base"]
            T[b + f"hc_{site}_scale.weight"] = p[site + "_hc_scale"]
        if i < HASH: T[b + "ffn_gate_tid2eid.weight"] = p["tid2eid"]
        else: T[b + "exp_probs_b.bias"] = p["probs_b"]
        if m:
            T.update({b + "attn_compressor_kv.weight": p["c_kv"], b + "attn_compressor_gate.weight": p["c_gate"],
                      b + "attn_compressor_ape.weight": p["c_ape"], b + "attn_compressor_norm.weight": p["c_norm"]})
            if m == 4:
                T.update({b + "indexer.attn_q_b.weight": p["i_q_b"], b + "indexer.proj.weight": p["i_proj"],
                          b + "indexer_compressor_kv.weight": p["ic_kv"], b + "indexer_compressor_gate.weight": p["ic_gate"],
                          b + "indexer_compressor_ape.weight": p["ic_ape"], b + "indexer_compressor_norm.weight": p["ic_norm"]})
    path = path or os.path.join(tempfile.mkdtemp(), "tiny-deepseek4.gguf")
    arch = "deepseek4"
    w = gguf.GGUFWriter(path, arch)
    w.add_block_count(L); w.add_context_length(CTX); w.add_embedding_length(E); w.add_vocab_size(V)
    w.add_head_count(H); w.add_head_count_kv(1); w.add_key_length(HD); w.add_value_length(HD)
    w.add_rope_dimension_count(ROT); w.add_rope_freq_base(THETA); w.add_layer_norm_rms_eps(EPS)
    w.add_string(f"{arch}.rope.scaling.type", "yarn"); w.add_float32(f"{arch}.rope.scaling.factor", YF)
    w.add_uint32(f"{arch}.rope.scaling.original_context_length", YORIG)
    w.add_float32(f"{arch}.rope.scaling.yarn_beta_fast", YBF); w.add_float32(f"{arch}.rope.scaling.yarn_beta_slow", YBS)
    w.add_uint32(f"{arch}.attention.q_lora_rank", QL); w.add_uint32(f"{arch}.attention.sliding_window", W)
    w.add_expert_count(NE); w.add_expert_used_count(NU); w.add_uint32(f"{arch}.expert_gating_func", 4)
    w.add_expert_feed_forward_length(FE); w.add_expert_shared_count(1)
    w.add_expert_weights_scale(WSCALE); w.add_expert_weights_norm(True)
    w.add_array(f"{arch}.swiglu_clamp_exp", [float(v) for v in CLAMP])
    w.add_array(f"{arch}.swiglu_clamp_shexp", [float(v) for v in CLAMP_S])
    w.add_uint32(f"{arch}.attention.indexer.head_count", IH); w.add_uint32(f"{arch}.attention.indexer.key_length", IHD)
    w.add_uint32(f"{arch}.attention.indexer.top_k", TOPK)
    w.add_uint32(f"{arch}.attention.output_group_count", OG); w.add_uint32(f"{arch}.attention.output_lora_rank", OL)
    w.add_array(f"{arch}.attention.compress_ratios", [int(v) for v in RATIOS] + [0])
    w.add_float32(f"{arch}.attention.compress_rope_freq_base", CTHETA)
    w.add_uint32(f"{arch}.hyper_connection.count", HC); w.add_uint32(f"{arch}.hyper_connection.sinkhorn_iterations", SINK_IT)
    w.add_float32(f"{arch}.hyper_connection.epsilon", HCEPS); w.add_uint32(f"{arch}.hash_layer_count", HASH)
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("default")
    w.add_token_list([f"t{i}" for i in range(V)]); w.add_token_types([1] * V); w.add_token_merges(["t 0", "t 1"])
    w.add_bos_token_id(1); w.add_eos_token_id(2)
    for name, t in T.items():
        w.add_tensor(name, np.ascontiguousarray(t, dtype=np.int32 if t.dtype == np.int32 else np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return path

# ---- NumPy reference (float64) ----
def rms(x, wt=None, eps=EPS):
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y if wt is None else y * wt
def sigmoid(x): return 1 / (1 + np.exp(-x))
def softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
def silu(x): return x * sigmoid(x)
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis, keepdims=True)); return e / e.sum(axis, keepdims=True)

def inv_freq(compressed):
    d = ROT
    ext = 1.0 / (CTHETA if compressed else THETA) ** (np.arange(0, d, 2) / d)
    if not compressed: return ext
    base = CTHETA
    corr = lambda nr: (d * math.log(YORIG / (nr * 2 * math.pi))) / (2 * math.log(base))
    low, high = max(math.floor(corr(YBF)), 0), min(math.ceil(corr(YBS)), d - 1)
    if low == high: high += 0.001
    ramp = np.clip((np.arange(d // 2) - low) / (high - low), 0, 1)
    extrap = 1 - ramp
    return (ext / YF) * (1 - extrap) + ext * extrap

def rope(x, pos, compressed, sign=1.0):     # interleaved pairs on the last ROT dims; x [..., n, d], pos [n]
    f = np.outer(pos, inv_freq(compressed))           # [n, ROT/2]
    cos, sin = np.cos(f), sign * np.sin(f)
    shape = (1,) * (x.ndim - 2) + (len(pos), ROT // 2)
    cos, sin = cos.reshape(shape), sin.reshape(shape)
    y = x.copy()
    a, b = x[..., -ROT::2], x[..., -ROT + 1::2]
    y[..., -ROT::2] = a * cos - b * sin
    y[..., -ROT + 1::2] = b * cos + a * sin
    return y

def hc_weights(p, site, X):                 # X [T, HC, E] -> pre [T,HC], post [T,HC], comb [T,HC,HC], collapsed [T,E]
    T_ = X.shape[0]
    mix = rms(X.reshape(T_, -1)) @ p[site + "_hc_fn"].T
    base, sc = p[site + "_hc_base"], p[site + "_hc_scale"]
    pre = sigmoid(mix[:, :HC] * sc[0] + base[:HC]) + HCEPS
    post = 2 * sigmoid(mix[:, HC:2 * HC] * sc[1] + base[HC:2 * HC])
    comb = softmax(mix[:, 2 * HC:].reshape(T_, HC, HC) * sc[2] + base[2 * HC:].reshape(HC, HC), -1) + HCEPS
    comb = comb / (comb.sum(-2, keepdims=True) + HCEPS)
    for _ in range(SINK_IT - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + HCEPS)
        comb = comb / (comb.sum(-2, keepdims=True) + HCEPS)
    collapsed = (pre[:, :, None] * X).sum(1)
    return pre, post, comb, collapsed

def compress(hs, kvw, gw, ape, normw, m, overlap, dim):   # all complete windows of the sequence -> entries [n, dim]
    T_ = hs.shape[0]
    nwin = T_ // m
    if nwin == 0: return np.zeros((0, dim))
    kv = (hs @ kvw.T)[:nwin * m].reshape(nwin, m, -1)
    g = (hs @ gw.T)[:nwin * m].reshape(nwin, m, -1) + ape
    if overlap:                                         # entry w: window w-1's first half with window w's second half
        nk = np.zeros((nwin, 2 * m, dim)); ng = np.full((nwin, 2 * m, dim), -np.inf)
        nk[:, m:] = kv[..., dim:]; ng[:, m:] = g[..., dim:]
        nk[1:, :m] = kv[:-1, :, :dim]; ng[1:, :m] = g[:-1, :, :dim]
        kv, g = nk, ng
    c = rms((kv * softmax(g, 1)).sum(1), normw)
    return rope(c, np.arange(nwin) * m, True)

def attention(p, i, hs, pos):
    T_, m = hs.shape[0], RATIOS[i]
    comp = m != 0
    qr = rms(hs @ p["q_a"].T, p["q_a_norm"])
    q = rope(rms((qr @ p["q_b"].T).reshape(T_, H, HD)).transpose(1, 0, 2), pos, comp)      # [H, T, HD]
    kv = rope(rms(hs @ p["kv"].T, p["kv_norm"])[None], pos, comp)[0]                         # [T, HD]
    keys = kv
    mask = np.where((pos[None, :] <= pos[:, None]) & (pos[None, :] > pos[:, None] - W), 0.0, -np.inf)  # [T, T]
    if comp:
        ce = compress(hs, p["c_kv"], p["c_gate"], p["c_ape"], p["c_norm"], m, m == 4, HD)
        thr = (pos + 1) // m
        n = ce.shape[0]
        bias = np.where(np.arange(n)[None, :] >= thr[:, None], -np.inf, 0.0)
        if m == 4 and n:
            ik = compress(hs, p["ic_kv"], p["ic_gate"], p["ic_ape"], p["ic_norm"], 4, True, IHD)
            iq = rope((qr @ p["i_q_b"].T).reshape(T_, IH, IHD).transpose(1, 0, 2), pos, True)   # [IH, T, IHD]
            sc = np.maximum(iq @ ik.T, 0) * IHD ** -0.5                                         # [IH, T, n]
            wts = (hs @ p["i_proj"].T) * IH ** -0.5                                            # [T, IH]
            score = (sc * wts.T[:, :, None]).sum(0)                                            # [T, n]
            score = np.where(bias == 0, score, -np.inf)
            k = min(TOPK, n)
            sel = np.full((T_, n), -np.inf)
            for t in range(T_):
                top = np.argsort(-score[t], kind="stable")[:k]
                top = top[top < thr[t]]
                sel[t, top] = 0.0
            bias = sel
        keys = np.concatenate([kv, ce], 0)
        mask = np.concatenate([mask, bias], 1)
    logits = q @ keys.T * HD ** -0.5 + mask[None]                                              # [H, T, K]
    sink = p["sinks"][:, None, None]
    mx = np.maximum(logits.max(-1, keepdims=True), sink)
    e = np.exp(logits - mx)
    w_ = e / (e.sum(-1, keepdims=True) + np.exp(sink - mx))
    out = rope(w_ @ keys, pos, comp, sign=-1.0)                                                # [H, T, HD]
    grouped = out.transpose(1, 0, 2).reshape(T_, OG, H * HD // OG)
    oa = p["o_a"].reshape(OG, OL, -1)
    y = np.einsum("tgi,goi->tgo", grouped, oa).reshape(T_, OG * OL)
    return y @ p["o_b"].T

PICKS = {}
def moe(p, i, x, ids):
    T_ = x.shape[0]
    s = np.sqrt(softplus(x @ p["router"].T))                                                  # [T, NE]
    if i < HASH: idx = p["tid2eid"][ids]
    else: idx = np.argsort(-(s + p["probs_b"]), 1, kind="stable")[:, :NU]
    PICKS[i] = [list(map(int, row)) for row in idx]
    wts = np.take_along_axis(s, idx, 1)
    wts = wts / (wts.sum(1, keepdims=True) + 1e-20) * WSCALE
    out = np.zeros_like(x)
    lim = CLAMP[i]
    for t in range(T_):
        for k in range(NU):
            e_ = idx[t, k]
            g = np.minimum(p["eg"][e_] @ x[t], lim); u = np.clip(p["eu"][e_] @ x[t], -lim, lim)
            out[t] += wts[t, k] * (p["ed"][e_] @ (silu(g) * u))
    ls = CLAMP_S[i]
    g = np.minimum(x @ p["sg"].T, ls); u = np.clip(x @ p["su"].T, -ls, ls)
    return out + (silu(g) * u) @ p["sd"].T

def reference(ids, all_rows=False):
    PICKS.clear()
    ids = np.asarray(ids)
    pos = np.arange(len(ids))
    X = np.repeat(P["embed"][ids].astype(np.float64)[:, None, :], HC, 1)                      # [T, HC, E]
    for i in range(L):
        p = P[i]
        pre, post, comb, col = hc_weights(p, "attn", X)
        a = attention(p, i, rms(col, p["attn_norm"]), pos)
        X = post[:, :, None] * a[:, None, :] + np.einsum("tkj,tke->tje", comb, X)
        pre, post, comb, col = hc_weights(p, "ffn", X)
        f = moe(p, i, rms(col, p["ffn_norm"]), ids)
        X = post[:, :, None] * f[:, None, :] + np.einsum("tkj,tke->tje", comb, X)
    T_ = X.shape[0]
    mix = rms(X.reshape(T_, -1)) @ P["hh_fn"].T
    pre = sigmoid(mix * P["hh_scale"][0] + P["hh_base"]) + HCEPS
    h = rms((pre[:, :, None] * X).sum(1), P["norm"])
    logits = h @ P["head"].T
    return logits if all_rows else logits[-1]

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--write":
        write_gguf(sys.argv[2]); sys.exit(0)
    exe = sys.argv[1]
    path = write_gguf()
    long_ids = list(np.random.default_rng(7).integers(0, V, 120))
    mid_ids = list(np.random.default_rng(8).integers(0, V, 37))
    ok = True
    exact = {"TIERLM_KV_F32": "1"}
    if "TIERLM_DEVICE" not in os.environ and not os.environ.get("TIERLM_RPC"): exact["TIERLM_DEVICE"] = "cpu"
    for label, env, tol in (("exact", exact, 1e-4), ("normal", {}, 6e-2)):
        worst_mode, all_top = {}, True
        # modes: whole prompt; one token at a time; saved half way and restored after another conversation;
        # a batch of 6 guesses checked and only some kept (as speculative decoding does), then one more token
        for mode in ([], ["--step"], ["snapshot"], ["--rollback"]):
            worst = 0.0
            for ids in ([5], [1, 2, 3], [7, 42, 91, 0, 13, 13, 64, 8, 21, 55], mid_ids, long_ids):
                args, menv, want_ids = list(mode), dict(env), list(ids)
                if mode == ["snapshot"]: args, menv["TIERLM_SNAPSHOT"] = [], "1"
                if mode == ["--rollback"]:
                    if len(ids) < 9: continue
                    keep = len(ids) % 4                     # 0 to 3 of the 6 kept
                    args = ["--rollback", "6", str(keep)]
                    want_ids = ids[:len(ids) - 7 + keep] + ids[-1:]
                want = reference(np.array(want_ids))
                run = subprocess.run([exe, path, *args, *map(str, ids)], capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", env={**os.environ, **menv})
                if run.returncode != 0: sys.exit(f"tierlm-logits failed: {run.stderr.strip()[-2000:]}")
                got = np.array([float(v) for v in run.stdout.split()])
                if len(got) != V: sys.exit(f"tierlm-logits printed {len(got)} numbers: {run.stderr.strip()[-2000:]}")
                worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
                all_top &= bool(got.argmax() == want.argmax())
            worst_mode[{"--step": "one token at a time", "snapshot": "saved and restored", "--rollback": "guesses undone"}.get(mode[0], "") if mode else "whole prompt"] = worst
        worst = max(worst_mode.values())
        passed = worst < tol and all_top
        ok &= passed
        detail = ", ".join(f"{k} {v:.1e}" for k, v in worst_mode.items())
        print(f"deepseek4 {label:7s}: worst relative diff {worst:.1e} (limit {tol:.0e}; {detail}), top token matches: {all_top}  ->  {'ok' if passed else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
