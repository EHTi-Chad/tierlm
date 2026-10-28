"""Correctness test for the Qwen3.5 / 3.6 / 3.8 hybrid architecture (GGUF arch "qwen35"), with no real model.

Builds a tiny random Qwen3.5 in Hugging Face's own parameterisation, writes it to GGUF the way the converter
does (norm weights stored as 1 + w, A_log stored as -exp(A_log), linear-attention value heads reordered from
grouped to tiled order, plus a multi-token-prediction block that must be ignored), then compares tierlm with a
plain NumPy implementation of Hugging Face's Qwen3_5 forward pass (Gated DeltaNet + gated attention).

usage: python tests/tiny_qwen35.py build/tierlm-logits
"""
import os, subprocess, sys, tempfile
import numpy as np
import gguf

rng = np.random.default_rng(35)
# Sizes are tiny but shaped like the real models where GPU kernels care: the CUDA DeltaNet kernel needs a head
# size of 16, 32, 64 or 128, its convolution a channel count that's a multiple of 128, and flash attention a
# head size of at least 64. (Real Qwen3.8-27B: head size 128 everywhere.)
V, E, H, KV, HD, FF, CTX = 120, 64, 2, 1, 64, 48, 512
ROT = HD // 4                          # partial rotary factor 0.25
HK, HV, S, KC = 2, 4, 16, 4            # linear attention: key heads, value heads, head size, conv width
L, INTERVAL, NEXTN, EPS, BASE = 8, 4, 1, 1e-6, 10000.0
CD = 2 * HK * S + HV * S
R = HV // HK
r = lambda *s: (rng.standard_normal(s) * 0.2).astype(np.float32)
small = lambda n: (rng.standard_normal(n) * 0.1).astype(np.float32)    # HF zero-centred norm weights
recurrent = [(i + 1) % INTERVAL != 0 for i in range(L)]

# ---- Hugging Face parameterisation ----
P = {"embed": r(V, E), "norm": small(E), "lm_head": r(V, E)}
for i in range(L):
    p = {"in_norm": small(E), "post_norm": small(E), "gate": r(FF, E), "up": r(FF, E), "down": r(E, FF)}
    if recurrent[i]:
        p.update({"qkv": r(CD, E), "z": r(HV * S, E), "b": r(HV, E), "a": r(HV, E), "conv": r(CD, KC),
                  "dt_bias": small(HV) + 0.5, "A_log": np.log(rng.uniform(0.5, 4, HV)).astype(np.float32),
                  "gnorm": (1 + small(S)).astype(np.float32), "out": r(E, HV * S)})
    else:
        p.update({"q": r(H * HD * 2, E), "k": r(KV * HD, E), "v": r(KV * HD, E), "o": r(E, H * HD),
                  "q_norm": small(HD), "k_norm": small(HD)})
    P[i] = p

# ---- write the GGUF the way the converter does ----
def tiled(x, axis, head):              # value heads: grouped by key head [g0v0 g0v1 g1v0 g1v1] -> tiled [g0v0 g1v0 g0v1 g1v1]
    x = np.moveaxis(x, axis, 0)
    sh = x.shape
    x = x.reshape(HK, R, head, *sh[1:]).transpose(1, 0, 2, *range(3, len(sh) + 2)).reshape(sh)
    return np.ascontiguousarray(np.moveaxis(x, 0, axis))

T = {"token_embd.weight": P["embed"], "output_norm.weight": 1 + P["norm"], "output.weight": P["lm_head"]}
for i in range(L):
    p, b = P[i], f"blk.{i}."
    T.update({b + "attn_norm.weight": 1 + p["in_norm"], b + "post_attention_norm.weight": 1 + p["post_norm"],
              b + "ffn_gate.weight": p["gate"], b + "ffn_up.weight": p["up"], b + "ffn_down.weight": p["down"]})
    if recurrent[i]:
        qk = 2 * HK * S
        T.update({b + "attn_qkv.weight": np.concatenate([p["qkv"][:qk], tiled(p["qkv"][qk:], 0, S)]),
                  b + "attn_gate.weight": tiled(p["z"], 0, S),
                  b + "ssm_beta.weight": tiled(p["b"], 0, 1), b + "ssm_alpha.weight": tiled(p["a"], 0, 1),
                  b + "ssm_conv1d.weight": np.concatenate([p["conv"][:qk], tiled(p["conv"][qk:], 0, S)]),
                  b + "ssm_dt.bias": tiled(p["dt_bias"], 0, 1), b + "ssm_a": tiled(-np.exp(p["A_log"]), 0, 1),
                  b + "ssm_norm.weight": p["gnorm"], b + "ssm_out.weight": tiled(p["out"], 1, S)})
    else:
        T.update({b + "attn_q.weight": p["q"], b + "attn_k.weight": p["k"], b + "attn_v.weight": p["v"],
                  b + "attn_output.weight": p["o"], b + "attn_q_norm.weight": 1 + p["q_norm"],
                  b + "attn_k_norm.weight": 1 + p["k_norm"]})
# a draft (multi-token prediction) block after the real layers: present in files, never run
b = f"blk.{L}."
T.update({b + "attn_norm.weight": np.full(E, 1e3, np.float32), b + "nextn.eh_proj.weight": r(E, 2 * E)})

path = os.path.join(tempfile.mkdtemp(), "tiny-qwen35.gguf")
w = gguf.GGUFWriter(path, "qwen35")
w.add_block_count(L + NEXTN); w.add_nextn_predict_layers(NEXTN)
w.add_embedding_length(E); w.add_head_count(H); w.add_head_count_kv(KV); w.add_key_length(HD); w.add_value_length(HD)
w.add_rope_dimension_count(ROT); w.add_rope_freq_base(BASE); w.add_layer_norm_rms_eps(EPS)
w.add_feed_forward_length(FF); w.add_context_length(CTX)
w.add_ssm_conv_kernel(KC); w.add_ssm_state_size(S); w.add_ssm_group_count(HK); w.add_ssm_time_step_rank(HV)
w.add_ssm_inner_size(HV * S); w.add_full_attention_interval(INTERVAL)
for name, t in T.items(): w.add_tensor(name, np.ascontiguousarray(t, dtype=np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

# ---- NumPy reference: Hugging Face's Qwen3_5 text model ----
def rms(x, wt, eps=EPS): return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * wt
def norm1(x, wt): return rms(x, 1 + wt)                      # Qwen3_5RMSNorm: weight is zero-centred
def silu(x): return x / (1 + np.exp(-x))
def softplus(x): return np.log1p(np.exp(x))
def softmax(x): x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)
def l2n(x): return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)

def rope(x, pos):                                            # partial NeoX rotation of the first ROT dims
    inv = BASE ** (-np.arange(0, ROT, 2) / ROT)
    ang = pos[:, None] * inv[None, :]
    cos = np.concatenate([np.cos(ang)] * 2, -1)[:, None, :]; sin = np.concatenate([np.sin(ang)] * 2, -1)[:, None, :]
    xr, xp = x[..., :ROT], x[..., ROT:]
    rot = np.concatenate([-xr[..., ROT // 2:], xr[..., :ROT // 2]], -1)
    return np.concatenate([xr * cos + rot * sin, xp], -1)

def deltanet(p, h):
    Tn = h.shape[0]
    mixed = h @ p["qkv"].T                                   # [T, CD]
    pad = np.concatenate([np.zeros((KC - 1, CD)), mixed])
    conv = np.stack([(pad[t:t + KC] * p["conv"].T).sum(0) for t in range(Tn)])
    conv = silu(conv)
    q = conv[:, :HK * S].reshape(Tn, HK, S); k = conv[:, HK * S:2 * HK * S].reshape(Tn, HK, S)
    v = conv[:, 2 * HK * S:].reshape(Tn, HV, S)
    z = (h @ p["z"].T).reshape(Tn, HV, S)
    beta = 1 / (1 + np.exp(-(h @ p["b"].T)))
    g = -np.exp(p["A_log"]) * softplus(h @ p["a"].T + p["dt_bias"])
    q, k = np.repeat(q, R, axis=1), np.repeat(k, R, axis=1)  # repeat_interleave: value head j uses key head j // R
    q, k = l2n(q) / np.sqrt(S), l2n(k)
    st = np.zeros((HV, S, S))
    out = np.zeros((Tn, HV, S))
    for t in range(Tn):
        st = st * np.exp(g[t])[:, None, None]
        kv_mem = (st * k[t][:, :, None]).sum(1)
        delta = (v[t] - kv_mem) * beta[t][:, None]
        st = st + k[t][:, :, None] * delta[:, None, :]
        out[t] = (st * q[t][:, :, None]).sum(1)
    out = rms(out, p["gnorm"]) * silu(z)
    return out.reshape(Tn, HV * S) @ p["out"].T

def attention(p, h, pos):
    Tn = h.shape[0]
    qg = (h @ p["q"].T).reshape(Tn, H, 2 * HD)
    q, gate = qg[..., :HD], qg[..., HD:].reshape(Tn, H * HD)
    q = norm1(q, p["q_norm"])
    k = norm1((h @ p["k"].T).reshape(Tn, KV, HD), p["k_norm"])
    v = (h @ p["v"].T).reshape(Tn, KV, HD)
    q, k = rope(q, pos), rope(k, pos)
    k, v = np.repeat(k, H // KV, axis=1), np.repeat(v, H // KV, axis=1)
    att = softmax(np.einsum("qhd,khd->hqk", q, k) / np.sqrt(HD) + np.triu(np.full((Tn, Tn), -np.inf), 1))
    o = np.einsum("hqk,khd->qhd", att, v).reshape(Tn, H * HD)
    return (o * (1 / (1 + np.exp(-gate)))) @ p["o"].T

def reference(ids):
    x = P["embed"][ids].astype(np.float64); pos = np.arange(len(ids), dtype=np.float64)
    for i in range(L):
        p = P[i]
        h = norm1(x, p["in_norm"])
        x = x + (deltanet(p, h) if recurrent[i] else attention(p, h, pos))
        h = norm1(x, p["post_norm"])
        x = x + (silu(h @ p["gate"].T) * (h @ p["up"].T)) @ p["down"].T
    return norm1(x, P["norm"])[-1] @ P["lm_head"].T


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
long_ids = list(np.random.default_rng(7).integers(0, V, 300))   # > 256: the running memory must carry across batches
ok = True
for label, env, tol in PASSES({}):
    worst_mode = {}
    all_top = True
    for mode in ([], ["--step"]):
        worst = 0.0
        for ids in ([5], [1, 2, 3], [7, 42, 99, 0, 13, 13, 64, 8, 21, 55], long_ids):
            want = reference(np.array(ids))
            run = subprocess.run([sys.argv[1], path, *mode, *map(str, ids)], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", env={**os.environ, **env})
            if run.returncode != 0:
                sys.exit(f"tierlm-logits failed: {run.stderr.strip()}")
            got = np.array([float(x) for x in run.stdout.split()])
            err = float(np.abs(got - want).max() / np.abs(want).max())
            if os.environ.get("TIERLM_VERBOSE"):
                print(f"   {'step ' if mode else 'whole'} {len(ids):3d} tokens: {err:.1e}")
            worst = max(worst, err)
            all_top &= bool(got.argmax() == want.argmax())
        worst_mode["one token at a time" if mode else "whole prompt"] = worst
    worst = max(worst_mode.values())
    passed = worst < tol and all_top
    ok &= passed
    detail = ", ".join(f"{k} {v:.1e}" for k, v in worst_mode.items())
    print(f"qwen35    {label:9s}: worst relative diff {worst:.1e} (limit {tol:.0e}; {detail}), top token matches: {all_top}  ->  {'ok' if passed else 'FAIL'}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
