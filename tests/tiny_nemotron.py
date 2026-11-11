"""Correctness test for NVIDIA's Nemotron 3.5 architecture (GGUF arch "nemotron_h_moe"), with no real model.

A Nemotron block is one of three things, each behind one RMS norm and added back to the stream:
  Mamba-2      a state-space mixer: the model's running memory (no attention, no positions)
  LatentMoE    experts that work in a narrower "latent" width, plus one shared expert at full width;
               blocks differ in how many experts they have, how wide those are and how many each token uses
  attention    plain causal attention with no position encoding (the Mamba blocks carry the order)

This builds a tiny random model with all three, writes it to GGUF with the names and layouts of NVIDIA's files,
and compares tierlm with a plain NumPy implementation of the model math (Mamba-2 as in the paper and in
state-spaces/mamba's Mamba2 module; the router as NVIDIA's NemotronH code does it).

usage: python tests/tiny_nemotron.py build/tierlm-logits
       python tests/tiny_nemotron.py --write out.gguf      (just writes the file; prints nothing else)
"""
import os, subprocess, sys, tempfile
import numpy as np
import gguf

rng = np.random.default_rng(74)
V, E, CTX, EPS = 96, 32, 512, 1e-5
# Mamba-2: inner width, heads (each HDM wide), state size, B/C groups, convolution width
DI, MH, DS, MG, KC = 64, 4, 8, 2, 4
HDM = DI // MH
CC = DI + 2 * MG * DS                  # channels through the convolution: x, then B and C for each group
DIN = 2 * DI + 2 * MG * DS + MH        # in_proj's rows: z | x B C | dt
# attention
H, KV, HD = 4, 2, 8
# LatentMoE: latent width, shared expert width, weights scale; per block: experts, expert width, experts per token
LAT, FS, WSCALE = 16, 24, 5.0
KINDS = ["M", "F", "M", "F", "A", "F", "M", "F", "M", "A"]
NE_L = {1: 6, 3: 8, 5: 5, 7: 8}
FE_L = {1: 16, 3: 12, 5: 16, 7: 12}
NU_L = {1: 2, 3: 3, 5: 2, 7: 2}
L = len(KINDS)
r = lambda *s: (rng.standard_normal(s) * 0.2).astype(np.float32)
nw = lambda n: (1 + rng.standard_normal(n) * 0.1).astype(np.float32)

P = {"embed": r(V, E), "norm": nw(E), "head": r(V, E)}
for i, kind in enumerate(KINDS):
    p = {"norm": nw(E)}
    if kind == "M":
        p.update({"in": r(DIN, E), "conv": r(CC, KC), "conv_b": r(CC), "dt_bias": r(MH) + 0.5,
                  "A_log": np.log(rng.uniform(0.5, 4, MH)).astype(np.float32), "D": (1 + r(MH)).astype(np.float32),
                  "gnorm": nw(DI), "out": r(E, DI),
                  # NVIDIA's 4-bit files store in/out as NVFP4 with one more number each, a scale on the result
                  "in_scale": np.float32(1.0 + 0.1 * i), "out_scale": np.float32(0.8 + 0.05 * i)})
    elif kind == "A":
        p.update({"q": r(H * HD, E), "k": r(KV * HD, E), "v": r(KV * HD, E), "o": r(E, H * HD)})
    else:
        ne, fe = NE_L[i], FE_L[i]
        p.update({"router": r(ne, E), "probs_b": r(ne) * 0.5, "lat_down": r(LAT, E), "lat_up": r(E, LAT),
                  "eu": r(ne, fe, LAT), "ed": r(ne, LAT, fe), "su": r(FS, E), "sd": r(E, FS)})
    P[i] = p

def write_gguf(path=None, scales=True):
    T = {"token_embd.weight": P["embed"], "output_norm.weight": P["norm"], "output.weight": P["head"]}
    for i, kind in enumerate(KINDS):
        p, b = P[i], f"blk.{i}."
        T[b + "attn_norm.weight"] = p["norm"]
        if kind == "M":
            T.update({b + "ssm_in.weight": p["in"], b + "ssm_conv1d.weight": p["conv"], b + "ssm_conv1d.bias": p["conv_b"],
                      b + "ssm_dt.bias": p["dt_bias"], b + "ssm_a": -np.exp(p["A_log"]).reshape(MH, 1),
                      b + "ssm_d": p["D"].reshape(MH, 1), b + "ssm_norm.weight": p["gnorm"].reshape(MG, DI // MG),
                      b + "ssm_out.weight": p["out"]})
            if scales:
                T.update({b + "ssm_in.scale": np.array([p["in_scale"]]), b + "ssm_out.scale": np.array([p["out_scale"]])})
        elif kind == "A":
            T.update({b + "attn_q.weight": p["q"], b + "attn_k.weight": p["k"], b + "attn_v.weight": p["v"],
                      b + "attn_output.weight": p["o"]})
        else:
            T.update({b + "ffn_gate_inp.weight": p["router"], b + "exp_probs_b.bias": p["probs_b"],
                      b + "ffn_latent_down.weight": p["lat_down"], b + "ffn_latent_up.weight": p["lat_up"],
                      b + "ffn_up_exps.weight": p["eu"], b + "ffn_down_exps.weight": p["ed"],
                      b + "ffn_up_shexp.weight": p["su"], b + "ffn_down_shexp.weight": p["sd"]})
    path = path or os.path.join(tempfile.mkdtemp(), "tiny-nemotron.gguf")
    arch = "nemotron_h_moe"
    w = gguf.GGUFWriter(path, arch)
    per = lambda f: [int(f(i)) for i in range(L)]
    w.add_block_count(L); w.add_context_length(CTX); w.add_embedding_length(E); w.add_vocab_size(V)
    w.add_array(f"{arch}.feed_forward_length", per(lambda i: FE_L.get(i, 0)))
    w.add_head_count(H); w.add_array(f"{arch}.attention.head_count_kv", per(lambda i: KV if KINDS[i] == "A" else 0))
    w.add_key_length(HD); w.add_value_length(HD); w.add_rope_dimension_count(HD)
    w.add_layer_norm_rms_eps(EPS); w.add_layer_norm_eps(EPS)
    w.add_expert_count(max(NE_L.values())); w.add_array(f"{arch}.expert_used_count", per(lambda i: NU_L.get(i, 0)))
    w.add_array(f"{arch}.expert_feed_forward_length", per(lambda i: FE_L.get(i, 0)))
    w.add_expert_shared_feed_forward_length(FS); w.add_expert_shared_count(1)
    w.add_expert_weights_norm(True); w.add_expert_weights_scale(WSCALE); w.add_uint32(f"{arch}.expert_gating_func", 2)
    w.add_uint32(f"{arch}.expert_group_count", 1); w.add_uint32(f"{arch}.expert_group_used_count", 1)
    w.add_uint32(f"{arch}.moe_latent_size", LAT)
    w.add_ssm_conv_kernel(KC); w.add_ssm_state_size(DS); w.add_ssm_group_count(MG); w.add_ssm_inner_size(DI)
    w.add_ssm_time_step_rank(MH)
    # a vocabulary, so other programs can open the file too (tierlm-logits doesn't need one)
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("default")
    w.add_token_list([f"t{i}" for i in range(V)]); w.add_token_types([1] * V); w.add_token_merges(["t 0", "t 1"])
    w.add_bos_token_id(1); w.add_eos_token_id(2)
    for name, t in T.items(): w.add_tensor(name, np.ascontiguousarray(t, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return path

# ---- NumPy reference ----
def rms(x, wt, eps=EPS): return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * wt
def silu(x): return x / (1 + np.exp(-x))
def sigmoid(x): return 1 / (1 + np.exp(-x))
def softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def mamba2(p, u):                                   # u: [T, E]
    T_ = u.shape[0]
    zxbcdt = (u @ p["in"].T) * p["in_scale"]
    z, xbc, dt = zxbcdt[:, :DI], zxbcdt[:, DI:DI + CC], zxbcdt[:, DI + CC:]
    # causal depthwise convolution: each channel's output at t is its last KC inputs times its KC taps, plus a bias
    pad = np.concatenate([np.zeros((KC - 1, CC), np.float64), xbc])
    conv = np.stack([(pad[t:t + KC] * p["conv"].T).sum(0) for t in range(T_)]) + p["conv_b"]
    xbc = silu(conv)
    x, B, C = xbc[:, :DI].reshape(T_, MH, HDM), xbc[:, DI:DI + MG * DS].reshape(T_, MG, DS), xbc[:, DI + MG * DS:].reshape(T_, MG, DS)
    dt = softplus(dt + p["dt_bias"])                # [T, MH]
    A = -np.exp(p["A_log"].astype(np.float64))
    state = np.zeros((MH, HDM, DS))
    y = np.zeros((T_, MH, HDM))
    for t in range(T_):
        for h in range(MH):
            g = h // (MH // MG)                     # each group's B and C serve MH / MG heads in a row
            state[h] = state[h] * np.exp(dt[t, h] * A[h]) + dt[t, h] * np.outer(x[t, h], B[t, g])
            y[t, h] = state[h] @ C[t, g] + p["D"][h] * x[t, h]
    y = y.reshape(T_, DI) * silu(z)
    y = rms(y.reshape(T_, MG, DI // MG), p["gnorm"].reshape(MG, DI // MG)).reshape(T_, DI)   # each group normed on its own
    return (y @ p["out"].T) * p["out_scale"]

def attention(p, u):
    T_ = u.shape[0]
    q = (u @ p["q"].T).reshape(T_, H, HD)
    k = np.repeat((u @ p["k"].T).reshape(T_, KV, HD), H // KV, axis=1)
    v = np.repeat((u @ p["v"].T).reshape(T_, KV, HD), H // KV, axis=1)
    out = np.zeros((T_, H, HD))
    for h in range(H):
        s = q[:, h] @ k[:, h].T / np.sqrt(HD)
        s = np.where(np.tril(np.ones((T_, T_))) > 0, s, -np.inf)
        s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
        out[:, h] = s @ v[:, h]
    return out.reshape(T_, H * HD) @ p["o"].T

PICKS = {}                                          # the last reference() run: block -> each token's experts, best first
def latent_moe(p, u, nu, i=-1):
    relu2 = lambda a: np.maximum(a, 0) ** 2
    shared = relu2(u @ p["su"].T) @ p["sd"].T
    probs = sigmoid(u @ p["router"].T)              # the router reads the full width
    lat = u @ p["lat_down"].T
    out = np.zeros((u.shape[0], LAT))
    for t in range(u.shape[0]):
        top = np.argsort(-(probs[t] + p["probs_b"]), kind="stable")[:nu]   # chosen with the bias, weighted without it
        PICKS.setdefault(i, []).append([int(e) for e in top])
        wt = probs[t, top]
        wt = wt / max(wt.sum(), 6.103515625e-5) * WSCALE
        for e, we in zip(top, wt):
            out[t] += we * (relu2(lat[t] @ p["eu"][e].T) @ p["ed"][e].T)
    return out @ p["lat_up"].T + shared

def reference(ids):
    PICKS.clear()
    x = P["embed"][ids].astype(np.float64)
    for i, kind in enumerate(KINDS):
        p = P[i]
        u = rms(x, p["norm"])
        x = x + (mamba2(p, u) if kind == "M" else attention(p, u) if kind == "A" else latent_moe(p, u, NU_L[i], i))
    return rms(x, P["norm"])[-1] @ P["head"].T

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--write":
        write_gguf(sys.argv[2]); sys.exit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "--oracle":    # compare this file's NumPy math with another program's logits
        path = write_gguf(scales=False)                      # (programs that read the scales only from NVFP4 tensors)
        for p in P.values():
            if isinstance(p, dict) and "in_scale" in p: p["in_scale"] = p["out_scale"] = np.float32(1.0)
        for ids in ([5], [1, 2, 3], [7, 42, 91, 0, 13, 13, 64, 8, 21, 55]):
            run = subprocess.run([sys.argv[2], path, *map(str, ids)], capture_output=True, text=True)
            got = np.array([float(v) for v in run.stdout.split()])
            want = reference(np.array(ids))
            print(len(ids), "tokens:", float(np.abs(got - want).max() / np.abs(want).max()) if len(got) == V else run.stderr[-400:])
        sys.exit(0)
    exe = sys.argv[1]
    path = write_gguf()
    long_ids = list(np.random.default_rng(7).integers(0, V, 300))
    SPLITS = [(mode, f) for mode in ("layers", "experts") for f in (0.0, 0.15, 0.4, 0.7, 2.0)] if os.environ.get("TIERLM_RPC") else [None]
    ok = True
    for split in SPLITS:
        split_env = {} if split is None else {"TIERLM_SPLIT": split[0], "TIERLM_VRAM_BYTES": str(int(os.path.getsize(path) * split[1]))}
        tag = "" if split is None else f" {split[0]}@{split[1]:.2f}"
        exact = {"TIERLM_KV_F32": "1", **split_env}
        if not os.environ.get("TIERLM_RPC") and "TIERLM_DEVICE" not in os.environ: exact["TIERLM_DEVICE"] = "cpu"
        # "written out": the Mamba recurrence as ordinary operations, token by token (what a reply's steps use when
        # the state is in VRAM), in place of ggml's scan; forced here for batches of up to 8 tokens
        for label, env, tol in (("exact", exact, 1e-4), ("written out", {**exact, "TIERLM_MAMBA_UNROLL": "8"}, 1e-4),
                                ("normal", dict(split_env), 6e-2)):
            worst_mode, all_top = {}, True
            for mode in ([], ["--step"], ["snapshot"]):
                worst = 0.0
                for ids in ([5], [1, 2, 3], [7, 42, 91, 0, 13, 13, 64, 8, 21, 55], long_ids):
                    want = reference(np.array(ids))
                    run = subprocess.run([exe, path, *[m for m in mode if m != "snapshot"], *map(str, ids)], capture_output=True, text=True,
                                         encoding="utf-8", errors="replace",
                                         env={**os.environ, **env, **({"TIERLM_SNAPSHOT": "1"} if mode == ["snapshot"] else {})})
                    if run.returncode != 0: sys.exit(f"tierlm-logits failed: {run.stderr.strip()}")
                    got = np.array([float(v) for v in run.stdout.split()])
                    worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
                    all_top &= bool(got.argmax() == want.argmax())
                worst_mode["one token at a time" if mode == ["--step"] else "saved and restored" if mode else "whole prompt"] = worst
            worst = 0.0
            for ids in ([7, 42, 91, 0, 13, 13, 64, 8, 21, 55], long_ids):
                run = subprocess.run([exe, path, "--parallel", "3", *map(str, ids)], capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", env={**os.environ, **env})
                if run.returncode != 0: sys.exit(f"tierlm-logits --parallel failed: {run.stderr.strip()}")
                got = np.array([float(v) for v in run.stdout.split()]).reshape(3, -1)
                n = len(ids)
                for sq in range(3):
                    want = reference(np.array([ids[(i + 3 * sq) % n] for i in range(max(1, n - sq))]))
                    worst = max(worst, float(np.abs(got[sq] - want).max() / np.abs(want).max()))
                    all_top &= bool(got[sq].argmax() == want.argmax())
            worst_mode["3 at once"] = worst
            # A check of guesses that is partly undone (speculative decoding): a batch of B tokens is evaluated as an
            # undoable step and only its first KEEP are accepted; the Mamba states go back to where they were then.
            worst = 0.0
            base = [7, 42, 91, 0, 13, 13, 64, 8, 21, 55, 3, 77]
            for B in (1, 3, 5):
                for keep in range(1, B + 1):       # (the first of a batch is never a guess)
                    want = reference(np.array(base[:len(base) - B - 1] + base[len(base) - B - 1:len(base) - 1][:keep] + base[-1:]))
                    run = subprocess.run([exe, path, "--rollback", str(B), str(keep), *map(str, base)], capture_output=True, text=True,
                                         encoding="utf-8", errors="replace", env={**os.environ, **env})
                    if run.returncode != 0: sys.exit(f"tierlm-logits --rollback failed: {run.stderr.strip()}")
                    got = np.array([float(v) for v in run.stdout.split()])
                    worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
                    all_top &= bool(got.argmax() == want.argmax())
            worst_mode["guesses undone"] = worst
            worst = max(worst_mode.values())
            passed = worst < tol and all_top
            ok &= passed
            detail = ", ".join(f"{k} {v:.1e}" for k, v in worst_mode.items())
            print(f"{'nemotron' + tag:22s}{label:11s}: worst relative diff {worst:.1e} (limit {tol:.0e}; {detail}), top token matches: {all_top}  ->  {'ok' if passed else 'FAIL'}")
    # The experts each token picks, as tierlm reports them (for the page's expert statistics, and later for keeping
    # the busiest in VRAM): blocks pick different numbers, the rest of a block's row is -1, as is a block without experts.
    ids = [7, 42, 91, 0, 13, 13, 64, 8, 21, 55]
    reference(np.array(ids))
    env = {**os.environ, "TIERLM_SHOW_ROUTING": "1", "TIERLM_KV_F32": "1"}
    if not os.environ.get("TIERLM_RPC") and "TIERLM_DEVICE" not in os.environ: env["TIERLM_DEVICE"] = "cpu"
    run = subprocess.run([exe, path, *map(str, ids)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    rows = [l[len("routing:"):].split("|") for l in run.stderr.splitlines() if l.startswith("routing:")]
    KMAX = max(NU_L.values())
    good = len(rows) == len(ids)
    for t_, row in enumerate(rows):
        for i in range(L):
            got = [int(v) for v in row[i].split()] if i < len(row) else []
            want = (PICKS[i][t_] + [-1] * KMAX)[:KMAX] if i in PICKS else [-1] * KMAX
            good &= got == want
    ok &= good
    print(f"nemotron  the experts picked, per block: {'ok' if good else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
