"""Correctness test for SenseNova-Loop (GGUF arch "sensenova_loop"), with no real model.

SenseNova-Loop is a Qwen3.5 (Gated DeltaNet + gated attention, see tiny_qwen35.py) whose layers [loop.start, loop.end)
run loop.count times a token, every pass with its own caches, and whose layers [mhc.start, mhc.end) keep the residual
stream as 2 copies mixed by manifold-constrained hyper-connections (mHC). Builds a tiny random one in Hugging Face's
parameterisation, writes it to GGUF the way OpenSenseNova's converter does (block_count counting the visits, each
layer's tensors under its first visit's number, the layers after the loop and the MTP block numbered by their visits,
q/k/v and the gate in one tensor, beta and alpha in one), and compares tierlm with a NumPy version of SenseNova's own
Hugging Face code (sensenova_loop/models/modeling_sensenova_loop.py: execution_plan, MHCRouter, enter/leave).
Variants: 2 passes with the mixing matrix made by 20 Sinkhorn steps (as released); 3 passes with mHC's fixed
identity mixing and a pre-scale.

usage: python tests/tiny_sensenova.py build/tierlm-logits
"""
import os, subprocess, sys, tempfile
import numpy as np
import gguf

rng = np.random.default_rng(57)
V, E, H, KV, HD, FF, CTX = 120, 64, 2, 1, 64, 48, 512
ROT = HD // 4
HK, HV, S, KC = 2, 4, 16, 4
L, INTERVAL, NEXTN, EPS, BASE = 12, 4, 1, 1e-6, 10000.0
CD = 2 * HK * S + HV * S
R = HV // HK
LO, HI, MLO, MHI, MS = 4, 8, 4, 8, 2
r = lambda *s: (rng.standard_normal(s) * 0.2).astype(np.float32)
small = lambda n: (rng.standard_normal(n) * 0.1).astype(np.float32)
recurrent = [(i + 1) % INTERVAL != 0 for i in range(L)]
VARIANTS = {"2 passes": dict(count=2, fixed=False, pre_scale=1.0, iters=20),
            "3 passes, fixed": dict(count=3, fixed=True, pre_scale=0.8, iters=20)}

P = {"embed": r(V, E), "norm": small(E), "lm_head": r(V, E)}
for i in range(L + NEXTN):
    p = {"in_norm": small(E), "post_norm": small(E), "gate": r(FF, E), "up": r(FF, E), "down": r(E, FF)}
    if i < L and recurrent[i]:
        p.update({"qkv": r(CD, E), "z": r(HV * S, E), "b": r(HV, E), "a": r(HV, E), "conv": r(CD, KC),
                  "dt_bias": small(HV) + 0.5, "A_log": np.log(rng.uniform(0.5, 4, HV)).astype(np.float32),
                  "gnorm": (1 + small(S)).astype(np.float32), "out": r(E, HV * S)})
    else:
        p.update({"q": r(H * HD * 2, E), "k": r(KV * HD, E), "v": r(KV * HD, E), "o": r(E, H * HD),
                  "q_norm": small(HD), "k_norm": small(HD)})
    if MLO <= i < MHI:
        # the routers: [pre (2) | post (2) | mixing logits (4)] (fixed identity: the first 4), initialised as SenseNova's
        # (pre 0.25 / 0.75, the mixing matrix's diagonal favoured) plus noise so every part matters
        for site in ("mhc_attn", "mhc_ffn"):
            bias = np.concatenate([np.log(np.array([0.25, 0.75]) / (1 - np.array([0.25, 0.75]))), rng.standard_normal(2) * 0.3,
                                   (np.eye(2) * np.log(15)).ravel() + rng.standard_normal(4)]).astype(np.float32)
            p[site] = ((rng.standard_normal((8, 2 * E)) * 0.3).astype(np.float32), bias)
    P[i] = p
P[L].update({"fc": r(E, 2 * E), "enorm": small(E), "hnorm": small(E), "mnorm": small(E)})

def tiled(x, axis, head):
    x = np.moveaxis(x, axis, 0)
    sh = x.shape
    x = x.reshape(HK, R, head, *sh[1:]).transpose(1, 0, 2, *range(3, len(sh) + 2)).reshape(sh)
    return np.ascontiguousarray(np.moveaxis(x, 0, axis))

def write_gguf(name, var):
    count = var["count"]
    extra = (count - 1) * (HI - LO)
    plan = list(range(LO)) + list(range(LO, HI)) * count + list(range(HI, L))
    first = {p: plan.index(p) for p in range(L)}
    first[L] = len(plan)                                         # the MTP block, after the visits
    T = {"token_embd.weight": P["embed"], "output_norm.weight": 1 + P["norm"], "output.weight": P["lm_head"]}
    for i in range(L + NEXTN):
        p, b = P[i], f"blk.{first[i]}."
        T.update({b + "attn_norm.weight": 1 + p["in_norm"], b + "post_attention_norm.weight": 1 + p["post_norm"],
                  b + "ffn_gate.weight": p["gate"], b + "ffn_up.weight": p["up"], b + "ffn_down.weight": p["down"]})
        if i < L and recurrent[i]:
            qk = 2 * HK * S
            T.update({b + "attn_qkv_gate.weight": np.concatenate([p["qkv"][:qk], tiled(p["qkv"][qk:], 0, S), tiled(p["z"], 0, S)]),
                      b + "ssm_beta_alpha.weight": np.concatenate([tiled(p["b"], 0, 1), tiled(p["a"], 0, 1)]),
                      b + "ssm_conv1d.weight": np.concatenate([p["conv"][:qk], tiled(p["conv"][qk:], 0, S)]),
                      b + "ssm_dt.bias": tiled(p["dt_bias"], 0, 1), b + "ssm_a": tiled(-np.exp(p["A_log"]), 0, 1),
                      b + "ssm_norm.weight": p["gnorm"], b + "ssm_out.weight": tiled(p["out"], 1, S)})
        else:
            T.update({b + "attn_q.weight": p["q"], b + "attn_k.weight": p["k"], b + "attn_v.weight": p["v"],
                      b + "attn_output.weight": p["o"], b + "attn_q_norm.weight": 1 + p["q_norm"],
                      b + "attn_k_norm.weight": 1 + p["k_norm"]})
        if MLO <= i < MHI:
            n = 4 if var["fixed"] else 8
            for site, key in (("mhc_attn", "hc_attn_fn"), ("mhc_ffn", "hc_ffn_fn")):
                w, bias = p[site]
                T.update({b + key + ".weight": w[:n], b + key + ".bias": bias[:n]})
    b, p = f"blk.{first[L]}.", P[L]
    T.update({b + "nextn.eh_proj.weight": p["fc"], b + "nextn.enorm.weight": 1 + p["enorm"],
              b + "nextn.hnorm.weight": 1 + p["hnorm"], b + "nextn.shared_head_norm.weight": 1 + p["mnorm"]})
    arch = "sensenova_loop"
    path = os.path.join(tempfile.mkdtemp(), "tiny-sensenova.gguf")
    w = gguf.GGUFWriter(path, arch)
    w.add_block_count(len(plan) + NEXTN); w.add_nextn_predict_layers(NEXTN)
    w.add_embedding_length(E); w.add_head_count(H); w.add_head_count_kv(KV); w.add_key_length(HD); w.add_value_length(HD)
    w.add_rope_dimension_count(ROT); w.add_rope_freq_base(BASE); w.add_layer_norm_rms_eps(EPS)
    w.add_feed_forward_length(FF); w.add_context_length(CTX)
    w.add_ssm_conv_kernel(KC); w.add_ssm_state_size(S); w.add_ssm_group_count(HK); w.add_ssm_time_step_rank(HV)
    w.add_ssm_inner_size(HV * S); w.add_full_attention_interval(INTERVAL)
    w.add_array(f"{arch}.attention.recurrent_layers", [recurrent[q] for q in plan] + [False])
    for k, v in (("loop.start", LO), ("loop.end", HI), ("loop.count", count), ("mhc.start", MLO), ("mhc.end", MHI),
                 ("mhc.streams", MS), ("mhc.iterations", var["iters"])):
        w.add_uint32(f"{arch}.{k}", v)
    w.add_bool(f"{arch}.mhc.fixed_identity", var["fixed"]); w.add_float32(f"{arch}.mhc.pre_scale", var["pre_scale"])
    for name_, t in T.items(): w.add_tensor(name_, np.ascontiguousarray(t, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return path

# ---- NumPy reference: Hugging Face's Qwen3_5 text model, run as SenseNova's execution plan with mHC ----
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

def ffn(i, h):
    p = P[i]
    return (silu(h @ p["gate"].T) * (h @ p["up"].T)) @ p["down"].T

def sigmoid(x): return 1 / (1 + np.exp(-x))

def sinkhorn(lg, iters):                                    # SenseNova's: rows then columns in log space, iters times
    for _ in range(iters):
        lg = lg - np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1, keepdims=True)) - lg.max(-1, keepdims=True)
        lg = lg - np.log(np.exp(lg - lg.max(-2, keepdims=True)).sum(-2, keepdims=True)) - lg.max(-2, keepdims=True)
    return np.exp(lg)

def router(Xs, site, var):                                  # MHCRouter.forward: Xs [T, MS, E]
    w, bias = site
    n = 4 if var["fixed"] else 8
    w, bias = w[:n], bias[:n]
    flat = Xs.reshape(Xs.shape[0], -1)
    lg = (flat / np.sqrt((flat * flat).mean(-1, keepdims=True) + 1e-6)) @ w.T + bias
    pre, post = sigmoid(lg[:, :MS]), 2 * sigmoid(lg[:, MS:2 * MS])
    mix = np.broadcast_to(np.eye(MS), (Xs.shape[0], MS, MS)) if var["fixed"] else sinkhorn(lg[:, 2 * MS:].reshape(-1, MS, MS), var["iters"])
    branch = np.einsum("ts,tse->te", pre, Xs) * var["pre_scale"]
    return branch, post, mix

def merge(Xs, y, post, mix): return np.einsum("trc,tce->tre", mix, Xs) + post[:, :, None] * y[:, None, :]

def hidden_states(ids, var):
    x = P["embed"][ids].astype(np.float64); pos = np.arange(len(ids), dtype=np.float64)
    plan = list(range(LO)) + list(range(LO, HI)) * var["count"] + list(range(HI, L))
    Xs = None
    for i in plan:
        p = P[i]
        block = (lambda h: deltanet(p, h)) if recurrent[i] else (lambda h: attention(p, h, pos))
        if MLO <= i < MHI:
            if i == MLO: Xs = np.repeat(x[:, None, :], MS, axis=1)
            br, post, mix = router(Xs, p["mhc_attn"], var)
            Xs = merge(Xs, block(norm1(br, p["in_norm"])), post, mix)
            br, post, mix = router(Xs, p["mhc_ffn"], var)
            Xs = merge(Xs, ffn(i, norm1(br, p["post_norm"])), post, mix)
            if i == MHI - 1: x = Xs.mean(1)
        else:
            x = x + block(norm1(x, p["in_norm"]))
            x = x + ffn(i, norm1(x, p["post_norm"]))
    return norm1(x, P["norm"])

def reference(ids, var): return hidden_states(ids, var)[-1] @ P["lm_head"].T

def mtp_reference(ids, var):
    Hs = hidden_states(ids, var)
    hprev = np.concatenate([np.zeros((1, E)), Hs[:-1]])
    p = P[L]; pos = np.arange(len(ids), dtype=np.float64)
    x = np.concatenate([norm1(P["embed"][ids].astype(np.float64), p["enorm"]), norm1(hprev, p["hnorm"])], -1) @ p["fc"].T
    x = x + attention(p, norm1(x, p["in_norm"]), pos)
    x = x + ffn(L, norm1(x, p["post_norm"]))
    return norm1(x, p["mnorm"])[-1] @ P["lm_head"].T

def PASSES(extra):
    exact_env = {"TIERLM_KV_F32": "1", **extra}
    if not os.environ.get("TIERLM_RPC") and "TIERLM_DEVICE" not in os.environ:
        exact_env["TIERLM_DEVICE"] = "cpu"
    passes = [("exact", exact_env, 1e-4, True), ("normal", dict(extra), 6e-2, True)]
    if not extra:
        # 8-bit KV cache: about 1% from exact. On these tiny random models the top two tokens are often nearly tied,
        # so only the size of the difference is checked.
        # (Here through 20 to 24 layer visits, mHC's mixing included: the cache's rounding adds up to about twice what it
        # does in tiny_qwen35.py, most in the shorter conversations of the 3-at-once check.)
        passes.append(("kv 8-bit", {"TIERLM_KV_TYPE": "q8_0", "TIERLM_DEVICE": os.environ.get("TIERLM_DEVICE", "cpu")}, 1.5e-1, False))
        # 5-bit: coarser (measured on these models: see the printed worst difference); a wrong cache layout would be 100%+
        passes.append(("kv 5-bit", {"TIERLM_KV_TYPE": "q5_0", "TIERLM_DEVICE": os.environ.get("TIERLM_DEVICE", "cpu")}, 6e-1, False))
    return passes
long_ids = list(np.random.default_rng(7).integers(0, V, 300))
SPLITS = [(mode, f) for mode in ("layers",) for f in (0.0, 0.3, 0.7, 2.0)] if os.environ.get("TIERLM_RPC") else [None]
ok = True
def run_logits(path, args, env):
    run = subprocess.run([sys.argv[1], path, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env={**os.environ, **env})
    if run.returncode != 0: sys.exit(f"tierlm-logits failed: {run.stderr.strip()[-3000:]}")
    return run
for name, var in VARIANTS.items():
    path = write_gguf(name, var)
    for split in SPLITS:
        split_env = {} if split is None else {"TIERLM_SPLIT": split[0], "TIERLM_VRAM_BYTES": str(int(os.path.getsize(path) * split[1]))}
        tag = "" if split is None else f" {split[0]}@{split[1]:.2f}"
        for label, env, tol, need_top in PASSES(split_env):
            worst_mode, all_top = {}, True
            for mode in ([], ["--step"], ["snapshot"]):
                worst = 0.0
                for ids in ([5], [1, 2, 3], [7, 42, 99, 0, 13, 13, 64, 8, 21, 55], long_ids):
                    want = reference(np.array(ids), var)
                    run = run_logits(path, [*[m for m in mode if m != "snapshot"], *map(str, ids)], {**env, **({"TIERLM_SNAPSHOT": "1"} if mode == ["snapshot"] else {})})
                    got = np.array([float(x) for x in run.stdout.split()])
                    worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
                    all_top &= bool(got.argmax() == want.argmax())
                worst_mode["one token at a time" if mode == ["--step"] else "saved and restored" if mode else "whole prompt"] = worst
            worst = 0.0
            for ids in ([7, 42, 99, 0, 13, 13, 64, 8, 21, 55], long_ids):
                got = np.array([float(x) for x in run_logits(path, ["--parallel", "3", *map(str, ids)], env).stdout.split()]).reshape(3, -1)
                n = len(ids)
                for sq in range(3):
                    want = reference(np.array([ids[(i + 3 * sq) % n] for i in range(max(1, n - sq))]), var)
                    worst = max(worst, float(np.abs(got[sq] - want).max() / np.abs(want).max()))
                    all_top &= bool(got[sq].argmax() == want.argmax())
            worst_mode["3 at once"] = worst
            worst = max(worst_mode.values())
            passed = worst < tol and (all_top or not need_top)
            ok &= passed
            detail = ", ".join(f"{k} {v:.1e}" for k, v in worst_mode.items())
            print(f"{name + tag:22s}{label:9s}: worst relative diff {worst:.1e} (limit {tol:.0e}; {detail}), top token matches: {all_top}  ->  {'ok' if passed else 'FAIL'}", flush=True)
        env = PASSES(split_env)[0][1]
        worst = 0.0
        for mode in (["--mtp"], ["--mtp-step"]):
            for ids in ([5], [1, 2, 3], [7, 42, 99, 0, 13, 13, 64, 8, 21, 55], long_ids):
                want = mtp_reference(np.array(ids), var)
                got = np.array([float(x) for x in run_logits(path, [*mode, *map(str, ids)], env).stdout.split()])
                worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
        passed = worst < 1e-4
        ok &= passed
        print(f"{name + tag:22s}{'MTP':9s}: worst relative diff {worst:.1e} (limit 1e-04)  ->  {'ok' if passed else 'FAIL'}", flush=True)
        # Speculative decoding (the MTP block; one pass of the loop; scripted guesses, partly wrong): the same greedy reply
        def gen(mode, extra=None):
            run = run_logits(path, [mode, "60", "5", "9", "13"], {**env, **(extra or {})})
            return run.stdout.split(), run.stderr.strip().splitlines()[-1] if run.stderr.strip() else ""
        plain, _ = gen("--gen")
        checks = [gen("--gen-spec")] + [gen("--gen-spec", {"TIERLM_NO_MTP": "1", "TIERLM_SELF_DRAFT": "1", "TIERLM_LOOP_DRAFT": n})
                                        for n in ("1", "2")]
        checks += [gen("--gen-spec", {"TIERLM_DRAFT_SCRIPT": ",".join(plain), "TIERLM_DRAFT_CORRUPT": k}) for k in ("3", "5")]
        same = all(c[0] == plain for c in checks)
        ok &= same
        print(f"{name + tag:22s}{'spec':9s}: greedy output with speculation {'identical' if same else 'DIFFERENT'} "
              f"({'; '.join(c[1] for c in checks)})  ->  {'ok' if same else 'FAIL'}", flush=True)
    os.remove(path)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
