"""Inkling 2 Mini (GGUF arch "inkling2_loop", text) against Hugging Face's own code.

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
three conversations at once, and with the sliding-window rings cut down to 48 tokens.

usage: python tests/tiny_inkling2.py build/tierlm-logits
"""
import os, subprocess, sys, tempfile
import numpy as np
import torch
import gguf
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
}

def build(over, seed):
    torch.manual_seed(seed)
    cfg = Inkling2LoopTextConfig(**{**BASE, **over})
    m = Inkling2LoopForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("norm.weight") or n.endswith("layernorm.weight") or n.endswith("norm.scale"):
                p.copy_(1 + 0.2 * torch.randn_like(p))
            elif n.endswith("mlp.gate.weight"): p.copy_(0.5 * torch.randn_like(p))      # (sigmoid scores: unsaturated, no ties)
            elif ".experts." in n: p.copy_(0.04 * torch.randn_like(p))          # (larger: tiny changes grow through the loop)
            elif n.endswith(".query"): p.copy_(0.3 * torch.randn_like(p))
            elif "message_gate" in n: p.copy_(0.2 * torch.randn_like(p))
            elif n.endswith("relative_position_bias.proj"): p.copy_(1.0 * torch.randn_like(p))
            else: p.copy_(0.12 * torch.randn_like(p))
        for n, b in m.named_buffers():
            if n.endswith("e_score_correction_bias"): b.copy_(0.3 * torch.randn_like(b))
    return cfg, m

def write(cfg, m):
    sd = {k: v.detach().float().numpy() for k, v in m.state_dict().items()}
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
        if lo <= i < hi:
            T[b + "loop_msg_query.weight"] = sd["model.message_input.query"][i - lo]
            for hf, gg in (("q", "q"), ("k", "k"), ("v", "v"), ("mlp", "ffn")):
                g = p + f"{hf}_message_gate."
                T[b + f"loop_{gg}_ctrl_down.weight"] = sd[g + "down_proj.weight"]
                T[b + f"loop_{gg}_ctrl_gain.weight"] = sd[g + "message_gate_proj.weight"]
                T[b + f"loop_{gg}_ctrl_forget.weight"] = sd[g + "forget_gate_proj.weight"]
    path = os.path.join(tempfile.mkdtemp(), "tiny-inkling2.gguf")
    A = "inkling2_loop"
    w = gguf.GGUFWriter(path, A)
    w.add_block_count(NL); w.add_context_length(cfg.max_position_embeddings)
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
    w.add_array(f"{A}.attention.sliding_window_pattern", swa)
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
    w.add_uint32(f"{A}.loop.start", cfg.loop_layer_index_start); w.add_uint32(f"{A}.loop.end", cfg.loop_layer_index_end)
    w.add_uint32(f"{A}.loop.count", cfg.loop_num_passes); w.add_uint32(f"{A}.loop.gate_rank", cfg.loop_gate_rank)
    w.add_bool(f"{A}.loop.varying", False)
    for name, t in T.items(): w.add_tensor(name, np.ascontiguousarray(t).astype(np.float32))
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
        cfg, m = build(over, 70 + ci)
        path = write(cfg, m)
        for label, env, tol in (("exact", {"TIERLM_KV_F32": "1", "TIERLM_DEVICE": "cpu"}, 1e-3), ("normal", {}, 6e-2),
                                ("ring48", {"TIERLM_KV_F32": "1", "TIERLM_DEVICE": "cpu", "TIERLM_SWA_RING_EXTRA": "48"}, 1e-3)):
            worst, top = 0.0, True
            for mode in ([], ["--step"], ["snapshot"]):
                for ids in tests:
                    want = ref_logits(m, ids)
                    r = subprocess.run([exe, path, *[x for x in mode if x != "snapshot"], *map(str, ids)], capture_output=True, text=True,
                                       env={**os.environ, **env, **({"TIERLM_SNAPSHOT": "1"} if mode == ["snapshot"] else {})})
                    if r.returncode != 0: sys.exit(f"{name}: tierlm-logits failed: {r.stderr.strip()[-3000:]}")
                    got = np.array([float(v) for v in r.stdout.split()])
                    if got.shape != want.shape: sys.exit(f"{name}: {got.shape[0]} logits, want {want.shape[0]}")
                    worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
                    top &= bool(got.argmax() == want.argmax())
            for ids in (tests[2], long_ids):                            # three conversations at once
                r = subprocess.run([exe, path, "--parallel", "3", *map(str, ids)], capture_output=True, text=True, env={**os.environ, **env})
                if r.returncode != 0: sys.exit(f"{name}: tierlm-logits --parallel failed: {r.stderr.strip()[-3000:]}")
                got = np.array([float(v) for v in r.stdout.split()]).reshape(3, -1)
                n = len(ids)
                for sq in range(3):
                    want = ref_logits(m, [ids[(i + 3 * sq) % n] for i in range(max(1, n - sq))])
                    worst = max(worst, float(np.abs(got[sq] - want).max() / np.abs(want).max()))
                    top &= bool(got[sq].argmax() == want.argmax())
            passed = worst < tol and top
            ok &= passed
            print(f"{name:8s} {label:7s}: worst relative diff {worst:.1e} (limit {tol:.0e}), top token matches: {top}  ->  {'ok' if passed else 'FAIL'}")
        os.remove(path)
    sys.exit(0 if ok else 1)
