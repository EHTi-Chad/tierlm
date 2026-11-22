"""Inkling 2 Mini (GGUF arch "inkling2_loop", or the earlier "inkling2" with num_loops; text) against Hugging Face's own code.

Builds tiny random models with transformers' Inkling2LoopForCausalLM and covers what the family is made of: no RoPE,
a relative-position bias read from each token's own "relative states" (clipped, and in the full-attention layers
scaled by log(position)), queries and keys clipped to a length, values normed, a sink logit per token and head, the
attention output gated; short causal convolutions on the keys, values and both blocks' outputs (in the full-attention
layers with taps that depend on the token too); attention over the depth instead of a plain residual (each layer
mixes the earlier blocks' outputs with learned queries); sigmoid-scored experts with a softcapped SwiGLU; logits
softcapped and the padded vocabulary dropped; and the loop: the middle layers run several times, each later pass
starting again from the same input, with messages from the previous pass mixed into its queries, keys, values and
experts' inputs. Each model is written as a GGUF laid out as llama.cpp's converter writes Inkling 2 (convolution taps
oldest first, the relative bias table [d_rel, extent] transposed, padded output rows), run with tierlm-logits and
compared with transformers' logits for the same tokens: whole prompts, token by token, with speculative snapshots,
three conversations at once, with the sliding-window rings cut down to 48 tokens, and with a prompt read in steps
that get smaller as the cache grows (what bounds the attention's working memory in a long context).

The "earlier" cases write the same models the way Unsloth's Inkling-2-Mini files are laid out: arch "inkling2" with
num_loops, the per-layer settings written out visit by visit (block_count counts the visits), the loop as
loop.core_start / loop.core_end, the gates named loop_{attn,k,v,mlp}_{down,gain,forget}, every loop layer's message
queries in two tensors (loop_res_query: each layer's q and mlp queries; loop_kv_res_query: its k and v queries), and
the experts with a scale each beside their weights (as the files' NVFP4 experts have: here the weights are divided
by a random scale per expert, which the scale tensors give back). "earlier-q" also stores the weights in the types
those files use: the matrices in BF16, the experts in NVFP4 with their scales (some layers' in BF16, without).

With TIERLM_RPC set (a tierlm-rpc-server standing in for a GPU) the cases run split between that device and RAM
instead: by layers and by experts at several VRAM budgets, and by experts also with the busiest experts in VRAM
(TIERLM_HOT=1) and with live ones.

usage: python tests/tiny_inkling2.py build/tierlm-logits [case...]
       TIERLM_RPC=127.0.0.1:50071 python tests/tiny_inkling2.py build-rpc/tierlm-logits [case...]
"""
import os, subprocess, sys, tempfile
import numpy as np
import torch
import gguf
import re
from gguf import quants, GGMLQuantizationType as Q
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM

V, VU = 300, 290                                                       # padded and real vocabulary
TYPES6 = ["hybrid_sliding", "hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid_sliding", "hybrid"]
BASE = dict(vocab_size=V, unpadded_vocab_size=VU, hidden_size=64, num_hidden_layers=6, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=16,
            sliding_window_size=8, d_rel=4, rel_extent=12, log_scaling_n_floor=4, log_scaling_alpha=0.3,
            n_layers_per_attn_res_block=2, moe_intermediate_size=24, n_routed_experts=8, num_experts_per_tok=2,
            loop_layer_index_start=2, loop_layer_index_end=4, loop_num_passes=3, loop_gate_rank=8, layer_types=TYPES6,
            eos_token_id=1, max_position_embeddings=4096, attn_implementation="eager")
CASES = {
    "loop3":   dict(),
    "loop2":   dict(loop_num_passes=2, n_layers_per_attn_res_block=1, loop_layer_index_start=1, loop_layer_index_end=5),
    "swa-gqa": dict(swa_num_attention_heads=6, swa_num_key_value_heads=3, swa_head_dim=8, sliding_window_size=5),
    "earlier":  dict(),
    "earlier2": dict(loop_num_passes=2, n_layers_per_attn_res_block=1, loop_layer_index_start=1, loop_layer_index_end=5,
                     swa_num_attention_heads=6, swa_num_key_value_heads=3, swa_head_dim=8),
    "earlier-q": dict(moe_intermediate_size=64),                       # (NVFP4 rows: 64 values to a block)
}
# "earlier-q": stored in the types the real files use, BF16 matrices and NVFP4 experts with their scales.
BF16 = re.compile(r"(token_embd|output|attn_(sink_proj|gate|k|q|v|output|rel|[kv]_conv_proj)|loop_\w+_(down|forget|gain)"
                  r"|ffn_(gate_up|down)_exps)\.weight$")

def bf16(x): return quants.dequantize(quants.quantize(np.ascontiguousarray(x, dtype=np.float32), Q.BF16), Q.BF16)

def nvfp4(x):
    """[..., n] (n a multiple of 64) as NVFP4: the type's bytes, and the values they hold. Blocks of 64 values in four
    groups of 16, each group with a scale (unsigned E4M3) and 4 bits a value (gguf has no writer for the type)."""
    N = quants.NVFP4
    s = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 4, 16)
    code = N.fp32_to_ue4m3(np.abs(s).max(-1) / 6)                      # (a group's largest: near the top value, 12 x scale / 2)
    kv = np.array(N.kvalues, np.float32)
    q = np.abs(s[..., None] - N.ue4m3_to_fp32(code)[..., None, None] * kv).argmin(-1).astype(np.uint8)
    qs = (q[..., :8] | (q[..., 8:] << 4)).reshape(-1, 32)              # value j and value j + 8 share a byte
    raw = np.concatenate([code.reshape(-1, 4).astype(np.uint8), qs], 1).reshape(*x.shape[:-1], -1)
    return raw, quants.dequantize(raw, Q.NVFP4)

def build(over, seed, expert_std=0.04):
    torch.manual_seed(seed)
    cfg = Inkling2LoopTextConfig(**{**BASE, **over})
    m = Inkling2LoopForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("norm.weight") or n.endswith("layernorm.weight") or n.endswith("norm.scale"):
                p.copy_(1 + 0.2 * torch.randn_like(p))
            elif n.endswith("mlp.gate.weight"): p.copy_(0.5 * torch.randn_like(p))      # (sigmoid scores: unsaturated, no ties)
            elif ".experts." in n: p.copy_(expert_std * torch.randn_like(p))          # (larger: tiny changes grow through the loop)
            elif n.endswith(".query"): p.copy_(0.3 * torch.randn_like(p))
            elif "message_gate" in n: p.copy_(0.2 * torch.randn_like(p))
            elif n.endswith("relative_position_bias.proj"): p.copy_(1.0 * torch.randn_like(p))
            else: p.copy_(0.12 * torch.randn_like(p))
        for n, b in m.named_buffers():
            if n.endswith("e_score_correction_bias"): b.copy_(0.3 * torch.randn_like(b))
    return cfg, m

def write(cfg, m, earlier=False, real_types=False):
    """real_types: the model's own weights are first rounded to what the file will hold, so the reference still is exact."""
    if real_types:
        with torch.no_grad():
            for n, p in m.named_parameters():
                if p.dim() >= 2: p.copy_(torch.from_numpy(bf16(p.numpy())))
    sd = {k: v.detach().float().numpy() for k, v in m.state_dict().items()}
    RAW = {}                                                           # tensors written in a type of their own
    rng = np.random.default_rng(11)
    NL, types = cfg.num_hidden_layers, cfg.layer_types
    swa = [t == "hybrid_sliding" for t in types]
    H = cfg.hidden_size
    out = np.zeros((V, H), dtype=np.float32)
    out[:VU] = sd["lm_head.weight"]                                    # (the checkpoint's padded rows: zeros)
    T = {"token_embd.weight": sd["model.embed_tokens.weight"],
         "token_embd_norm.weight": sd["model.embed_tokens.embed_norm.weight"],
         "output.weight": out, "output_norm.weight": sd["model.norm.weight"],
         "output_res_query.weight": sd["model.final_input.query"]}
    heads, kvh = [], []
    for i in range(NL):
        p, b = f"model.layers.{i}.", f"blk.{i}."
        a = p + "self_attn."
        heads.append(cfg.swa_num_attention_heads if swa[i] else cfg.num_attention_heads)
        kvh.append(cfg.swa_num_key_value_heads if swa[i] else cfg.num_key_value_heads)
        T.update({b + "attn_q.weight": sd[a + "q_proj.linear.weight"],
                  b + "attn_q_clip.weight": sd[a + "q_proj.norm.scale"],
                  b + "attn_k.weight": sd[a + "k_proj.linear.weight"],
                  b + "attn_v.weight": sd[a + "v_proj.linear.weight"],
                  b + "attn_output.weight": sd[a + "o_proj.linear.weight"],
                  b + "attn_gate.weight": sd[a + "o_proj.gate_proj.weight"],
                  b + "attn_sink_proj.weight": sd[a + "sink_proj.weight"],
                  b + "attn_rel.weight": sd[a + "relative_position_bias.linear.weight"],
                  b + "attn_rel_bias.weight": sd[a + "relative_position_bias.proj"].T,   # [extent, d_rel]
                  b + "attn_rel_clip.weight": sd[a + "relative_position_bias.norm.scale"],
                  b + "attn_post_conv.weight": sd[p + "post_attention_convolution.conv1d.weight"][:, 0, :],
                  b + "ffn_post_conv.weight": sd[p + "post_mlp_convolution.conv1d.weight"][:, 0, :],
                  b + "ffn_res_query.weight": sd[p + "mlp_input.query"],
                  b + "ffn_norm.weight": sd[p + "post_attention_layernorm.weight"],
                  b + "ffn_gate_inp.weight": sd[p + "mlp.gate.weight"],
                  b + "exp_probs_b.bias": sd[p + "mlp.gate.e_score_correction_bias"],
                  b + "ffn_gate_up_exps.weight": sd[p + "mlp.experts.gate_up_proj"],
                  b + "ffn_down_exps.weight": sd[p + "mlp.experts.down_proj"]})
        for kv in ("k", "v"):
            if swa[i]:
                T[b + f"attn_{kv}_conv.weight"] = sd[a + f"{kv}_proj.sconv.conv1d.weight"][:, 0, :]
            else:                                                       # static taps newest first -> oldest first
                T[b + f"attn_{kv}_conv.weight"] = np.ascontiguousarray(sd[a + f"{kv}_proj.sconv.static_weight"][::-1].T)
                T[b + f"attn_{kv}_conv_proj.weight"] = sd[a + f"{kv}_proj.sconv.weight_proj.weight"]
        if swa[i]: T[b + "attn_norm.weight"] = sd[p + "input_layernorm.weight"]
        else:
            for x in "qkv": T[b + f"attn_{x}_inp_norm.weight"] = sd[p + f"{x}_input_layernorm.weight"]
        if i > 0: T[b + "attn_res_query.weight"] = sd[p + "attention_input.query"]
        lo, hi = cfg.loop_layer_index_start, cfg.loop_layer_index_end
        if earlier and i % 3 != 2:                                      # (most layers, as the real files: not all)
            for nm, hf in (("ffn_gate_up_exps", "gate_up_proj"), ("ffn_down_exps", "down_proj")):
                sc = rng.uniform(0.3, 3.0, size=cfg.n_routed_experts).astype(np.float32)
                T[b + nm + ".weight"] = T[b + nm + ".weight"] / sc[:, None, None]
                T[b + nm + ".scale"] = sc
                if real_types:
                    raw, held = nvfp4(T[b + nm + ".weight"])
                    RAW[b + nm + ".weight"] = (raw, Q.NVFP4)
                    with torch.no_grad(): m.state_dict()[p + "mlp.experts." + hf].copy_(torch.from_numpy(held * sc[:, None, None]))
        if lo <= i < hi and earlier:
            for hf, gg in (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
                g = p + f"{hf}_message_gate."
                T[b + f"loop_{gg}_down.weight"] = sd[g + "down_proj.weight"]
                T[b + f"loop_{gg}_gain.weight"] = sd[g + "message_gate_proj.weight"]
                T[b + f"loop_{gg}_forget.weight"] = sd[g + "forget_gate_proj.weight"]
        elif lo <= i < hi:
            T[b + "loop_msg_query.weight"] = sd["model.message_input.query"][i - lo]
            for hf, gg in (("q", "q"), ("k", "k"), ("v", "v"), ("mlp", "ffn")):
                g = p + f"{hf}_message_gate."
                T[b + f"loop_{gg}_ctrl_down.weight"] = sd[g + "down_proj.weight"]
                T[b + f"loop_{gg}_ctrl_gain.weight"] = sd[g + "message_gate_proj.weight"]
                T[b + f"loop_{gg}_ctrl_forget.weight"] = sd[g + "forget_gate_proj.weight"]
    lo, hi = cfg.loop_layer_index_start, cfg.loop_layer_index_end
    if earlier:
        mq = sd["model.message_input.query"]                           # [loop layers, 4 (q, k, v, mlp), H]
        T["loop_res_query.weight"] = np.concatenate([np.stack([mq[i][0], mq[i][3]]) for i in range(hi - lo)])
        T["loop_kv_res_query.weight"] = np.concatenate([np.stack([mq[i][1], mq[i][2]]) for i in range(hi - lo)])
        # the settings that go by layer, written out visit by visit: before the loop, each pass, after
        visits = list(range(lo)) + list(range(lo, hi)) * cfg.loop_num_passes + list(range(hi, NL))
        heads, kvh, swa_w = [heads[v] for v in visits], [kvh[v] for v in visits], [swa[v] for v in visits]
    else: swa_w = swa
    path = os.path.join(tempfile.mkdtemp(), "tiny-inkling2.gguf")
    A = "inkling2" if earlier else "inkling2_loop"
    w = gguf.GGUFWriter(path, A)
    w.add_block_count(len(heads)); w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(H); w.add_feed_forward_length(cfg.moe_intermediate_size)
    w.add_array(f"{A}.attention.head_count", heads); w.add_array(f"{A}.attention.head_count_kv", kvh)
    w.add_key_length(cfg.head_dim); w.add_value_length(cfg.head_dim)
    w.add_uint32(f"{A}.attention.key_length_swa", cfg.swa_head_dim); w.add_uint32(f"{A}.attention.value_length_swa", cfg.swa_head_dim)
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    w.add_expert_count(cfg.n_routed_experts); w.add_expert_used_count(cfg.num_experts_per_tok)
    w.add_uint32(f"{A}.expert_feed_forward_length", cfg.moe_intermediate_size)
    w.add_float32(f"{A}.expert_weights_scale", cfg.route_scale)
    w.add_uint32(f"{A}.vocab_size", V); w.add_uint32(f"{A}.vocab_size_unpadded", VU)
    w.add_uint32(f"{A}.rope.dimension_count", 0)
    w.add_uint32(f"{A}.attention.sliding_window", cfg.sliding_window_size)
    w.add_array(f"{A}.attention.sliding_window_pattern", swa_w)
    w.add_uint32(f"{A}.shortconv.l_cache", cfg.conv_kernel_size)
    w.add_float32(f"{A}.activation.situ_beta", cfg.mlp_gate_softcap)
    w.add_float32(f"{A}.activation.situ_linear_beta", cfg.mlp_up_softcap)
    w.add_uint32(f"{A}.attn_res.block_size", cfg.n_layers_per_attn_res_block)
    w.add_float32(f"{A}.final_logit_softcapping", cfg.final_logit_softcapping)
    w.add_uint32(f"{A}.attention.relative_length", cfg.d_rel)
    w.add_uint32(f"{A}.attention.relative_extent", cfg.rel_extent)
    w.add_float32(f"{A}.attention.qk_clip", cfg.qk_l2_clip_threshold)
    w.add_float32(f"{A}.attention.relative_clip", cfg.rel_l2_clip_threshold)
    w.add_uint32(f"{A}.attention.log_scale_floor", cfg.log_scaling_n_floor)
    w.add_float32(f"{A}.attention.log_scale_alpha", cfg.log_scaling_alpha)
    if earlier:
        w.add_uint32(f"{A}.loop.core_start", lo); w.add_uint32(f"{A}.loop.core_end", hi)
        w.add_uint32(f"{A}.num_loops", cfg.loop_num_passes); w.add_uint32(f"{A}.loop.gate_rank", cfg.loop_gate_rank)
    else:
        w.add_uint32(f"{A}.loop.start", lo); w.add_uint32(f"{A}.loop.end", hi)
        w.add_uint32(f"{A}.loop.count", cfg.loop_num_passes); w.add_uint32(f"{A}.loop.gate_rank", cfg.loop_gate_rank)
        w.add_bool(f"{A}.loop.varying", False)
    for name, t in T.items():
        if name in RAW: w.add_tensor(name, RAW[name][0], raw_dtype=RAW[name][1])
        elif real_types and BF16.search(name): w.add_tensor(name, quants.quantize(np.ascontiguousarray(t, dtype=np.float32), Q.BF16), raw_dtype=Q.BF16)
        else: w.add_tensor(name, np.ascontiguousarray(t).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return path

def ref_logits(m, ids):
    with torch.no_grad():
        return m(torch.tensor([ids])).logits[0, -1].double().numpy()

if __name__ == "__main__":
    exe = sys.argv[1]
    only = sys.argv[2:]
    long_ids = [int(x) for x in np.random.default_rng(3).integers(2, VU, 300)]   # > 256: batched prompts
    tests = [[5], [1, 2, 3], [7, 42, 99, 0, 13, 13, 64, 8, 21, 55, 117, 4, 4, 9], long_ids]
    ok = True
    for ci, (name, over) in enumerate(CASES.items()):
        if only and name not in only: continue
        # "earlier-q": weights in BF16 and NVFP4 are multiplied with what they read rounded as well (to BF16, to 8 bits),
        # which the reference does not do; a few thousandths a matrix, and in a model this small now and then an expert
        # picked differently. So that case goes by the typical difference over the prompts, with a loose cap on the
        # worst: a weight type read wrongly gives differences near 1 everywhere.
        real = name == "earlier-q"
        cfg, m = build(over, 70 + ci, 0.025 if real else 0.04)
        path = write(cfg, m, earlier=name.startswith("earlier"), real_types=real)
        if os.environ.get("TIERLM_RPC"):                                # split between the stand-in device and RAM
            size = os.path.getsize(path)
            for split, frac, hot in [(sp, f, h) for sp in ("layers", "experts") for f in (0.0, 0.15, 0.4, 0.7, 2.0)
                                     for h in ((None,) if sp == "layers" else (None, "1", "live"))]:
                env = {**os.environ, "TIERLM_KV_F32": "1", "TIERLM_SPLIT": split, "TIERLM_VRAM_BYTES": str(int(size * frac)),
                       "TIERLM_SHOW_PLACEMENT": "1", **({"TIERLM_HOT": hot} if hot else {}), **({"TIERLM_LIVE_EAGER": "1"} if hot == "live" else {})}
                diffs, place = [], ""
                for mode in ([], ["--step"]):
                    for ids in tests:
                        want = ref_logits(m, ids)
                        r = subprocess.run([exe, path, *mode, *map(str, ids)], capture_output=True, text=True, env=env)
                        if r.returncode != 0: sys.exit(f"{name} {split} {frac} {hot}: tierlm-logits failed ({r.returncode}): {r.stderr.strip()[-3000:]}")
                        place = next((l for l in r.stderr.splitlines() if "VRAM" in l), "")
                        got = np.array([float(v) for v in r.stdout.split()])
                        diffs.append(float(np.abs(got - want).max() / np.abs(want).max()))
                mid, worst = float(np.median(diffs)), max(diffs)
                passed = (mid < 6e-2 and worst < 0.6) if real else worst < 1e-3
                ok &= passed
                print(f"{name:9s} {split:7s} {frac:4.2f} {'busiest' if hot == '1' else hot or '':7s}: typical relative diff {mid:.1e}, worst {worst:.1e}"
                      f"  ->  {'ok' if passed else 'FAIL'}  [{place.split(': ', 1)[-1]}]", flush=True)
            os.remove(path)
            continue
        for label, env, tol in (("exact", {"TIERLM_KV_F32": "1", "TIERLM_DEVICE": "cpu"}, 6e-2 if real else 1e-3), ("normal", {}, 6e-2),
                                ("ring48", {"TIERLM_KV_F32": "1", "TIERLM_DEVICE": "cpu", "TIERLM_SWA_RING_EXTRA": "48"}, 6e-2 if real else 1e-3),
                                # (a prompt read in steps that shrink as the cache grows, as in a long context: 64 tokens, then 32)
                                ("steps", {"TIERLM_KV_F32": "1", "TIERLM_DEVICE": "cpu", "TIERLM_BATCH_SPAN": "64"}, 6e-2 if real else 1e-3)):
            worst, top, diffs = 0.0, True, []
            for mode in ([], ["--step"], ["snapshot"]):
                for ids in tests:
                    want = ref_logits(m, ids)
                    r = subprocess.run([exe, path, *[x for x in mode if x != "snapshot"], *map(str, ids)], capture_output=True, text=True,
                                       env={**os.environ, **env, **({"TIERLM_SNAPSHOT": "1"} if mode == ["snapshot"] else {})})
                    if r.returncode != 0: sys.exit(f"{name}: tierlm-logits failed: {r.stderr.strip()[-3000:]}")
                    got = np.array([float(v) for v in r.stdout.split()])
                    if got.shape != want.shape: sys.exit(f"{name}: {got.shape[0]} logits, want {want.shape[0]}")
                    diffs.append(float(np.abs(got - want).max() / np.abs(want).max()))
                    top &= bool(got.argmax() == want.argmax())
            for ids in (tests[2], long_ids):                            # three conversations at once
                r = subprocess.run([exe, path, "--parallel", "3", *map(str, ids)], capture_output=True, text=True, env={**os.environ, **env})
                if r.returncode != 0: sys.exit(f"{name}: tierlm-logits --parallel failed: {r.stderr.strip()[-3000:]}")
                got = np.array([float(v) for v in r.stdout.split()]).reshape(3, -1)
                n = len(ids)
                for sq in range(3):
                    want = ref_logits(m, [ids[(i + 3 * sq) % n] for i in range(max(1, n - sq))])
                    diffs.append(float(np.abs(got[sq] - want).max() / np.abs(want).max()))
                    top &= bool(got[sq].argmax() == want.argmax())
            worst = max(diffs)
            if real:
                mid = float(np.median(diffs))
                passed = mid < tol and worst < 0.6
                print(f"{name:9s} {label:7s}: typical relative diff {mid:.1e} (limit {tol:.0e}), worst {worst:.1e} (limit 6e-01)  ->  {'ok' if passed else 'FAIL'}")
            else:
                passed = worst < tol and top
                print(f"{name:9s} {label:7s}: worst relative diff {worst:.1e} (limit {tol:.0e}), top token matches: {top}  ->  {'ok' if passed else 'FAIL'}")
            ok &= passed
        os.remove(path)
    sys.exit(0 if ok else 1)
