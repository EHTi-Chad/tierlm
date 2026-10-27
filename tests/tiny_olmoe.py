"""Correctness test for the OLMoE graph, using no real model and almost no memory.

Builds a tiny OLMoE with random weights, writes it as a GGUF, computes the logits
with a plain NumPy reference (Hugging Face OLMoE semantics), runs tierlm-logits on
the same file and compares the two.

usage: python tests/tiny_olmoe.py build/tierlm-logits
"""
import os, subprocess, sys, tempfile
import numpy as np
import gguf

rng = np.random.default_rng(0)
L, E, H, FF, NE, NU, V, CTX, BASE, EPS = 2, 64, 4, 32, 8, 2, 100, 128, 10000.0, 1e-5
HD = E // H
r = lambda *s: (rng.standard_normal(s) * 0.2).astype(np.float32)
nw = lambda: (1.0 + rng.standard_normal(E) * 0.1).astype(np.float32)

W = {"token_embd.weight": r(V, E), "output_norm.weight": nw(), "output.weight": r(V, E)}
for i in range(L):
    b = f"blk.{i}."
    W.update({b + "attn_norm.weight": nw(), b + "attn_q.weight": r(E, E), b + "attn_k.weight": r(E, E),
              b + "attn_v.weight": r(E, E), b + "attn_output.weight": r(E, E),
              b + "attn_q_norm.weight": nw(), b + "attn_k_norm.weight": nw(), b + "ffn_norm.weight": nw(),
              b + "ffn_gate_inp.weight": r(NE, E), b + "ffn_gate_exps.weight": r(NE, FF, E),
              b + "ffn_up_exps.weight": r(NE, FF, E), b + "ffn_down_exps.weight": r(NE, E, FF)})

path = os.path.join(tempfile.mkdtemp(), "tiny-olmoe.gguf")
w = gguf.GGUFWriter(path, "olmoe")
w.add_block_count(L); w.add_embedding_length(E); w.add_head_count(H); w.add_head_count_kv(H)
w.add_feed_forward_length(FF); w.add_expert_count(NE); w.add_expert_used_count(NU)
w.add_context_length(CTX); w.add_rope_freq_base(BASE); w.add_layer_norm_rms_eps(EPS)
for name, t in W.items(): w.add_tensor(name, t)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

def rms(x, g): return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * g
def softmax(x): x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)
def rope(x, pos):                      # x: [T, H, HD], GPT-NeoX style (rotate halves)
    inv = BASE ** (-np.arange(0, HD, 2) / HD)
    ang = pos[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., :HD // 2], x[..., HD // 2:]
    return np.concatenate([a * c - b * s, b * c + a * s], -1)

def reference(ids):
    T = len(ids); pos = np.arange(T, dtype=np.float64)
    x = W["token_embd.weight"][ids].astype(np.float64)
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(L):
        b = f"blk.{i}."
        h = rms(x, W[b + "attn_norm.weight"])
        q = rms(h @ W[b + "attn_q.weight"].T, W[b + "attn_q_norm.weight"]).reshape(T, H, HD)
        k = rms(h @ W[b + "attn_k.weight"].T, W[b + "attn_k_norm.weight"]).reshape(T, H, HD)
        v = (h @ W[b + "attn_v.weight"].T).reshape(T, H, HD)
        q, k = rope(q, pos), rope(k, pos)
        att = softmax(np.einsum("qhd,khd->hqk", q, k) / np.sqrt(HD) + mask)
        o = np.einsum("hqk,khd->qhd", att, v).reshape(T, E)
        x = x + o @ W[b + "attn_output.weight"].T
        h = rms(x, W[b + "ffn_norm.weight"])
        p = softmax(h @ W[b + "ffn_gate_inp.weight"].T)
        out = np.zeros_like(x)
        for t in range(T):
            for e in np.argsort(-p[t])[:NU]:   # OLMoE: top-k of softmax, no renormalising
                g = W[b + "ffn_gate_exps.weight"][e] @ h[t]
                u = W[b + "ffn_up_exps.weight"][e] @ h[t]
                out[t] += p[t, e] * (W[b + "ffn_down_exps.weight"][e] @ (g / (1 + np.exp(-g)) * u))
        x = x + out
    x = rms(x, W["output_norm.weight"])
    return x[-1] @ W["output.weight"].T

worst = 0.0
for mode in ([], ["--step"]):               # all at once, then one token per call through the KV cache
    label = "one call " if not mode else "KV cache "
    for ids in ([5], [1, 2, 3], [7, 42, 99, 0, 13, 13, 64, 8, 21, 55]):
        want = reference(np.array(ids))
        run = subprocess.run([sys.argv[1], path, *mode, *map(str, ids)], capture_output=True, text=True)
        if run.returncode != 0:
            sys.exit(f"tierlm-logits failed: {run.stderr.strip()}")
        got = np.array([float(v) for v in run.stdout.split()])
        err = float(np.abs(got - want).max())
        worst = max(worst, err)
        print(f"{label}{len(ids):2d} tokens: max |diff| {err:.2e}  top token ours {got.argmax()} / reference {want.argmax()}")
print("PASS" if worst < 1e-3 else "FAIL")
