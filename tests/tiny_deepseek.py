"""DeepSeek V2/V3, Kimi K2 and Moonlight (GGUF arch "deepseek2") against Hugging Face's own code.

Builds tiny random DeepSeek models with transformers' DeepseekV3ForCausalLM (multi-head latent attention, a dense
first layer, a shared expert, sigmoid routing with a choosing bias, expert groups, YaRN), writes each as a GGUF in
the layouts real files use (one attn_kv_b tensor; attn_k_b / attn_v_b as tierlm uses them; attn_k_b / attn_v_b
transposed), runs tierlm-logits on it and compares the logits with transformers' for the same tokens.

usage: python tests/tiny_deepseek.py build/tierlm-logits
"""
import os, subprocess, sys, tempfile
import numpy as np
import torch
import gguf
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM, DeepseekV3Config, DeepseekV3ForCausalLM

V = 120
CASES = {
    # name: config overrides, gguf layout
    "v3":          (dict(), "split"),
    "v3-kv_b":     (dict(), "kv_b"),
    "v3-transp":   (dict(), "transposed"),
    "no-q-lora":   (dict(q_lora_rank=None), "split"),                       # DeepSeek-V2-Lite: plain query projection
    "groups":      (dict(n_group=4, topk_group=2), "split"),                # DeepSeek V3: experts chosen from the best groups
    "yarn":        (dict(rope_parameters=dict(rope_type="yarn", rope_theta=10000.0, factor=8.0, original_max_position_embeddings=64,
                                              beta_fast=32.0, beta_slow=1.0, mscale=1.0, mscale_all_dim=1.0)), "split"),
    # DeepSeek V2 / V2-Lite: softmax routing, no renormalising, experts chosen from the groups with the best expert
    "v2":          (dict(v2=True, q_lora_rank=None, topk_method="group_limited_greedy", n_group=4, topk_group=2), "kv_b"),
    "v2-lite":     (dict(v2=True, q_lora_rank=None, topk_method="greedy", n_group=1, topk_group=1, routed_scaling_factor=1.0), "kv_b"),
}

def build(over, seed):
    torch.manual_seed(seed)
    base = dict(vocab_size=V, hidden_size=64, intermediate_size=96, moe_intermediate_size=24, num_hidden_layers=3,
                num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=16, num_experts_per_tok=4,
                routed_scaling_factor=2.5, kv_lora_rank=32, q_lora_rank=24, qk_rope_head_dim=8, qk_nope_head_dim=16,
                v_head_dim=12, n_group=1, topk_group=1, first_k_dense_replace=1, norm_topk_prob=True,
                max_position_embeddings=512, rms_norm_eps=1e-6, tie_word_embeddings=False, rope_interleave=True,
                rope_parameters=dict(rope_type="default", rope_theta=10000.0), attn_implementation="eager")
    v2 = over.pop("v2", False) if isinstance(over, dict) else False
    base.update(over)
    if v2:
        base = {k: v for k, v in base.items() if k not in ("norm_topk_prob", "rope_interleave")}
        cfg = DeepseekV2Config(**base); m = DeepseekV2ForCausalLM(cfg).eval()
    else:
        cfg = DeepseekV3Config(**base); m = DeepseekV3ForCausalLM(cfg).eval()
    with torch.no_grad():                                  # random weights big enough to matter, and a real choosing bias
        for n, p in m.named_parameters():
            if "norm" in n or "layernorm" in n: p.copy_(1 + 0.1 * torch.randn_like(p))
            else: p.copy_(0.2 * torch.randn_like(p))
        for l in m.model.layers:
            if hasattr(l.mlp, "gate") and hasattr(l.mlp.gate, "e_score_correction_bias"): l.mlp.gate.e_score_correction_bias.copy_(0.3 * torch.randn_like(l.mlp.gate.e_score_correction_bias))
    return cfg, m

def write(cfg, m, layout):
    sd = {k: v.detach().float().numpy() for k, v in m.state_dict().items()}
    H, R, N, P, Vd = cfg.num_attention_heads, cfg.kv_lora_rank, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
    W = {"token_embd.weight": sd["model.embed_tokens.weight"], "output_norm.weight": sd["model.norm.weight"],
         "output.weight": sd["lm_head.weight"]}
    for i in range(cfg.num_hidden_layers):
        p, b = f"model.layers.{i}.", f"blk.{i}."
        a = p + "self_attn."
        W[b + "attn_norm.weight"] = sd[p + "input_layernorm.weight"]
        W[b + "ffn_norm.weight"] = sd[p + "post_attention_layernorm.weight"]
        if cfg.q_lora_rank:
            W[b + "attn_q_a.weight"] = sd[a + "q_a_proj.weight"]; W[b + "attn_q_a_norm.weight"] = sd[a + "q_a_layernorm.weight"]
            W[b + "attn_q_b.weight"] = sd[a + "q_b_proj.weight"]
        else:
            W[b + "attn_q.weight"] = sd[a + "q_proj.weight"]
        W[b + "attn_kv_a_mqa.weight"] = sd[a + "kv_a_proj_with_mqa.weight"]
        W[b + "attn_kv_a_norm.weight"] = sd[a + "kv_a_layernorm.weight"]
        kvb = sd[a + "kv_b_proj.weight"]                    # [H*(N+Vd), R]
        per = kvb.reshape(H, N + Vd, R)
        if layout == "kv_b":
            W[b + "attn_kv_b.weight"] = kvb
        elif layout == "split":                             # ggml ne: k_b [N, R, H], v_b [R, Vd, H]
            W[b + "attn_k_b.weight"] = np.ascontiguousarray(per[:, :N, :].transpose(0, 2, 1))
            W[b + "attn_v_b.weight"] = np.ascontiguousarray(per[:, N:, :])
        else:                                               # transposed: k_b [R, N, H], v_b [Vd, R, H]
            W[b + "attn_k_b.weight"] = np.ascontiguousarray(per[:, :N, :])
            W[b + "attn_v_b.weight"] = np.ascontiguousarray(per[:, N:, :].transpose(0, 2, 1))
        W[b + "attn_output.weight"] = sd[a + "o_proj.weight"]
        mp = p + "mlp."
        if i < cfg.first_k_dense_replace:
            W[b + "ffn_gate.weight"] = sd[mp + "gate_proj.weight"]; W[b + "ffn_up.weight"] = sd[mp + "up_proj.weight"]
            W[b + "ffn_down.weight"] = sd[mp + "down_proj.weight"]
        else:
            W[b + "ffn_gate_inp.weight"] = sd[mp + "gate.weight"]
            if mp + "gate.e_score_correction_bias" in sd: W[b + "exp_probs_b.bias"] = sd[mp + "gate.e_score_correction_bias"]
            gu = sd[mp + "experts.gate_up_proj"]            # [E, 2F, Hd]
            F = gu.shape[1] // 2
            W[b + "ffn_gate_exps.weight"] = np.ascontiguousarray(gu[:, :F, :])
            W[b + "ffn_up_exps.weight"] = np.ascontiguousarray(gu[:, F:, :])
            W[b + "ffn_down_exps.weight"] = sd[mp + "experts.down_proj"]
            W[b + "ffn_gate_shexp.weight"] = sd[mp + "shared_experts.gate_proj.weight"]
            W[b + "ffn_up_shexp.weight"] = sd[mp + "shared_experts.up_proj.weight"]
            W[b + "ffn_down_shexp.weight"] = sd[mp + "shared_experts.down_proj.weight"]
    path = os.path.join(tempfile.mkdtemp(), f"tiny-deepseek-{layout}.gguf")
    w = gguf.GGUFWriter(path, "deepseek2")
    A = "deepseek2"
    w.add_block_count(cfg.num_hidden_layers); w.add_embedding_length(cfg.hidden_size); w.add_feed_forward_length(cfg.intermediate_size)
    w.add_head_count(H); w.add_head_count_kv(H); w.add_context_length(cfg.max_position_embeddings)
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps); w.add_rope_freq_base(cfg.rope_parameters["rope_theta"])
    w.add_key_length(N + P); w.add_value_length(Vd); w.add_rope_dimension_count(P)
    if cfg.q_lora_rank: w.add_uint32(f"{A}.attention.q_lora_rank", cfg.q_lora_rank)
    w.add_uint32(f"{A}.attention.kv_lora_rank", R)
    w.add_expert_count(cfg.n_routed_experts); w.add_expert_used_count(cfg.num_experts_per_tok)
    w.add_expert_feed_forward_length(cfg.moe_intermediate_size); w.add_uint32(f"{A}.expert_shared_count", cfg.n_shared_experts)
    w.add_uint32(f"{A}.leading_dense_block_count", cfg.first_k_dense_replace)
    v2 = isinstance(cfg, DeepseekV2Config)
    w.add_float32(f"{A}.expert_weights_scale", cfg.routed_scaling_factor); w.add_bool(f"{A}.expert_weights_norm", False if v2 else cfg.norm_topk_prob)
    w.add_uint32(f"{A}.expert_gating_func", 1 if v2 else 2)  # softmax (V2), sigmoid (V3)
    w.add_uint32(f"{A}.expert_group_count", cfg.n_group); w.add_uint32(f"{A}.expert_group_used_count", cfg.topk_group)
    rp = cfg.rope_parameters
    if rp.get("rope_type") == "yarn":
        w.add_rope_scaling_type(gguf.RopeScalingType.YARN); w.add_rope_scaling_factor(rp["factor"])
        w.add_rope_scaling_orig_ctx_len(rp["original_max_position_embeddings"])
        w.add_float32(f"{A}.rope.scaling.yarn_log_multiplier", 0.1 * rp["mscale_all_dim"])
    for name, t in W.items(): w.add_tensor(name, t.astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return path

def ref_logits(m, ids):
    with torch.no_grad():
        return m(torch.tensor([ids])).logits[0, -1].double().numpy()

exe = sys.argv[1]
long_ids = [int(x) for x in np.random.default_rng(3).integers(0, V, 300)]
tests = [[5], [1, 2, 3], [7, 42, 99, 0, 13, 13, 64, 8, 21, 55, 117], long_ids]
ok = True
for ci, (name, (over, layout)) in enumerate(CASES.items()):
    cfg, m = build(over, 10 + ci)
    path = write(cfg, m, layout)
    for label, env, tol in (("exact", {"TIERLM_KV_F32": "1", "TIERLM_DEVICE": "cpu"}, 1e-4),
                            ("exact-nofa", {"TIERLM_KV_F32": "1", "TIERLM_DEVICE": "cpu", "TIERLM_NO_FA": "1"}, 1e-4),
                            ("normal", {}, 6e-2)):
        worst, top = 0.0, True
        for mode in ([], ["--step"], ["snapshot"]):        # snapshot: saved half way, restored after another conversation
            for ids in tests:
                want = ref_logits(m, ids)
                r = subprocess.run([exe, path, *[x for x in mode if x != "snapshot"], *map(str, ids)], capture_output=True, text=True,
                                   env={**os.environ, **env, **({"TIERLM_SNAPSHOT": "1"} if mode == ["snapshot"] else {})})
                if r.returncode != 0: sys.exit(f"{name}: tierlm-logits failed: {r.stderr.strip()[-600:]}")
                got = np.array([float(v) for v in r.stdout.split()])
                worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
                top &= bool(got.argmax() == want.argmax())
        for ids in (tests[2], long_ids):                    # three conversations at once, each against the reference
            r = subprocess.run([exe, path, "--parallel", "3", *map(str, ids)], capture_output=True, text=True, env={**os.environ, **env})
            if r.returncode != 0: sys.exit(f"{name}: tierlm-logits --parallel failed: {r.stderr.strip()[-600:]}")
            got = np.array([float(v) for v in r.stdout.split()]).reshape(3, -1)
            n = len(ids)
            for sq in range(3):
                want = ref_logits(m, [ids[(i + 3 * sq) % n] for i in range(max(1, n - sq))])
                worst = max(worst, float(np.abs(got[sq] - want).max() / np.abs(want).max()))
                top &= bool(got[sq].argmax() == want.argmax())
        passed = worst < tol and top
        ok &= passed
        print(f"{name:10s} {label:10s}: worst relative diff {worst:.1e} (limit {tol:.0e}), top token matches: {top}  ->  {'ok' if passed else 'FAIL'}")
    os.remove(path)
sys.exit(0 if ok else 1)
