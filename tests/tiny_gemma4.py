"""Gemma 4 (GGUF arch "gemma4", text) against Hugging Face's own code.

Builds tiny random Gemma 4 text models with transformers' Gemma4ForCausalLM and covers what the family adds over
Gemma 3: full-attention layers with their own (wider) head size, fewer key/value heads, keys reused as values and
proportional RoPE (only the first quarter of each head rotated); values normed without a weight; attention scale 1;
the last layers reading an earlier layer's cache (with twice as wide an MLP); per-layer inputs (a second token table
plus a projection of the embedding, fed into every layer); a scale on each layer's output; and the MoE block (a dense
MLP beside 8 experts, each path with its own norms, the router reading the un-normed residual, a scale per expert).
Each is written as a GGUF laid out as llama.cpp's converter writes Gemma 4 (norm weights as they are, a rope_freqs
tensor for the proportional RoPE), run with tierlm-logits and compared with transformers' logits for the same tokens.

usage: python tests/tiny_gemma4.py build/tierlm-logits
"""
import os, subprocess, sys, tempfile
import numpy as np
import torch
import gguf
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

V = 300
BASE = dict(vocab_size=V, hidden_size=64, intermediate_size=96, num_hidden_layers=7, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, global_head_dim=32, num_global_key_value_heads=1, attention_k_eq_v=True,
            sliding_window=8, num_kv_shared_layers=0, hidden_size_per_layer_input=0, max_position_embeddings=1024,
            rms_norm_eps=1e-6, final_logit_softcapping=None, attn_implementation="eager")
CASES = {
    "dense":       dict(),                                               # (as the 31B)
    "separate-v":  dict(attention_k_eq_v=False, num_global_key_value_heads=None),
    "shared-ple":  dict(num_hidden_layers=10, num_kv_shared_layers=3, hidden_size_per_layer_input=8,   # (as E2B / E4B)
                        vocab_size_per_layer_input=V, use_double_wide_mlp=True, attention_k_eq_v=False,
                        num_global_key_value_heads=None),
    "moe":         dict(enable_moe_block=True, num_experts=8, top_k_experts=2, moe_intermediate_size=24),   # (as 26B A4B)
    "softcap":     dict(final_logit_softcapping=30.0),
}

def build(over, seed):
    torch.manual_seed(seed)
    cfg = Gemma4TextConfig(**{**BASE, **over})
    m = Gemma4ForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n or n.endswith("router.scale") or n.endswith("per_expert_scale"): p.copy_(1 + 0.1 * torch.randn_like(p))
            elif n.endswith("router.proj.weight"): p.copy_(2.0 * torch.randn_like(p))   # clear picks: no near-ties to flip
            else: p.copy_(0.2 * torch.randn_like(p))
        for layer in m.model.layers: layer.layer_scalar.copy_(0.7 + 0.6 * torch.rand(1))
    return cfg, m

def write(cfg, m):
    sd = {k: v.detach().float().numpy() for k, v in m.state_dict().items()}
    NL, types = cfg.num_hidden_layers, cfg.layer_types
    plc = [cfg.per_layer_config[i] for i in range(NL)]                 # (head size and key/value heads differ by layer)
    shd = next(c.head_dim for c, t in zip(plc, types) if t == "sliding_attention")
    ghd = next(c.head_dim for c, t in zip(plc, types) if t == "full_attention")
    swa = [t == "sliding_attention" for t in types]
    T = {"token_embd.weight": sd["model.embed_tokens.weight"], "output_norm.weight": sd["model.norm.weight"]}
    if not m.config.tie_word_embeddings: T["output.weight"] = sd["lm_head.weight"]
    full = cfg.rope_parameters["full_attention"]
    rot = int(full.get("partial_rotary_factor", 1.0) * ghd // 2)
    T["rope_freqs.weight"] = np.array([1.0] * rot + [1e30] * (ghd // 2 - rot), dtype=np.float32)
    P = cfg.hidden_size_per_layer_input
    if P:
        T["per_layer_token_embd.weight"] = sd["model.embed_tokens_per_layer.weight"]
        T["per_layer_model_proj.weight"] = sd["model.per_layer_model_projection.weight"]
        T["per_layer_proj_norm.weight"] = sd["model.per_layer_projection_norm.weight"]
    kvh = []
    for i in range(NL):
        p, b = f"model.layers.{i}.", f"blk.{i}."
        a, f = p + "self_attn.", p + "mlp."
        kvh.append(plc[i].num_key_value_heads)
        T.update({b + "attn_norm.weight": sd[p + "input_layernorm.weight"],
                  b + "attn_q.weight": sd[a + "q_proj.weight"], b + "attn_output.weight": sd[a + "o_proj.weight"],
                  b + "attn_q_norm.weight": sd[a + "q_norm.weight"],
                  b + "post_attention_norm.weight": sd[p + "post_attention_layernorm.weight"],
                  b + "ffn_norm.weight": sd[p + "pre_feedforward_layernorm.weight"],
                  b + "post_ffw_norm.weight": sd[p + "post_feedforward_layernorm.weight"],
                  b + "ffn_gate.weight": sd[f + "gate_proj.weight"], b + "ffn_up.weight": sd[f + "up_proj.weight"],
                  b + "ffn_down.weight": sd[f + "down_proj.weight"],
                  b + "layer_output_scale.weight": sd[p + "layer_scalar"]})
        for hf, gg in (("k_proj", "attn_k"), ("v_proj", "attn_v"), ("k_norm", "attn_k_norm")):
            if a + hf + ".weight" in sd: T[b + gg + ".weight"] = sd[a + hf + ".weight"]
        if P:
            T.update({b + "inp_gate.weight": sd[p + "per_layer_input_gate.weight"],
                      b + "proj.weight": sd[p + "per_layer_projection.weight"],
                      b + "post_norm.weight": sd[p + "post_per_layer_input_norm.weight"]})
        if cfg.enable_moe_block:
            T.update({b + "ffn_gate_inp.weight": sd[p + "router.proj.weight"], b + "ffn_gate_inp.scale": sd[p + "router.scale"],
                      b + "ffn_gate_up_exps.weight": sd[p + "experts.gate_up_proj"],
                      b + "ffn_down_exps.weight": sd[p + "experts.down_proj"],
                      b + "ffn_down_exps.scale": sd[p + "router.per_expert_scale"],
                      b + "pre_ffw_norm_2.weight": sd[p + "pre_feedforward_layernorm_2.weight"],
                      b + "post_ffw_norm_1.weight": sd[p + "post_feedforward_layernorm_1.weight"],
                      b + "post_ffw_norm_2.weight": sd[p + "post_feedforward_layernorm_2.weight"]})
    path = os.path.join(tempfile.mkdtemp(), "tiny-gemma4.gguf")
    w = gguf.GGUFWriter(path, "gemma4"); A = "gemma4"
    w.add_block_count(NL); w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(cfg.hidden_size); w.add_feed_forward_length(cfg.intermediate_size)
    w.add_head_count(cfg.num_attention_heads); w.add_head_count_kv(kvh)
    w.add_key_length(ghd); w.add_value_length(ghd)
    w.add_uint32(f"{A}.attention.key_length_swa", shd); w.add_uint32(f"{A}.attention.value_length_swa", shd)
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps); w.add_sliding_window(cfg.sliding_window)
    w.add_array(f"{A}.attention.sliding_window_pattern", swa)
    w.add_rope_freq_base(full["rope_theta"])
    w.add_float32(f"{A}.rope.freq_base_swa", cfg.rope_parameters["sliding_attention"]["rope_theta"])
    w.add_rope_dimension_count(ghd); w.add_uint32(f"{A}.rope.dimension_count_swa", shd)
    if cfg.num_kv_shared_layers: w.add_uint32(f"{A}.attention.shared_kv_layers", cfg.num_kv_shared_layers)
    if P: w.add_uint32(f"{A}.embedding_length_per_layer_input", P)
    if cfg.enable_moe_block:
        w.add_expert_count(cfg.num_experts); w.add_expert_used_count(cfg.top_k_experts)
        w.add_expert_feed_forward_length(cfg.moe_intermediate_size)
    if cfg.final_logit_softcapping: w.add_float32(f"{A}.final_logit_softcapping", cfg.final_logit_softcapping)
    for name, t in T.items(): w.add_tensor(name, np.ascontiguousarray(t).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return path

def ref_logits(m, ids):
    with torch.no_grad():
        return m(torch.tensor([ids])).logits[0, -1].double().numpy()

# ggml's CPU computes GELU from a table of 16-bit values: the exact pass's reference does the same (see tiny_gemma3.py).
# Gemma 4 has up to three GELUs a layer (the MLP, the experts, the per-layer input), and a value a hair from a rounding
# step can land on either side of it: 1e-3 here (a wrong weight or a missing step is 1e-2 or more).
class GeluTable(torch.nn.Module):
    def forward(self, x):
        g = torch.nn.functional.gelu(x.half().float(), approximate="tanh").half().float()
        return torch.where(x <= -10, torch.zeros_like(x), torch.where(x >= 10, x, g))
gelu_table = GeluTable()
def set_act(m, fn):
    for layer in m.model.layers:
        layer.mlp.act_fn = fn
        if hasattr(layer, "act_fn"): layer.act_fn = fn
        if hasattr(layer, "experts"): layer.experts.act_fn = fn

exe = sys.argv[1]
only = sys.argv[2:]
long_ids = [int(x) for x in np.random.default_rng(3).integers(2, V, 300)]   # > 256: batched prompts
tests = [[5], [1, 2, 3], [7, 42, 99, 0, 13, 13, 64, 8, 21, 55, 117, 4, 4, 9], long_ids]
ok = True
for ci, (name, over) in enumerate(CASES.items()):
    if only and name not in only: continue
    cfg, m = build(over, 50 + ci)
    path = write(cfg, m)
    exact_act = m.model.layers[0].mlp.act_fn
    for label, env, tol in (("exact", {"TIERLM_KV_F32": "1", "TIERLM_DEVICE": "cpu"}, 1e-3), ("normal", {}, 6e-2)):
        set_act(m, gelu_table if label == "exact" else exact_act)
        worst, top = 0.0, True
        for mode in ([], ["--step"], ["snapshot"]):
            for ids in tests:
                want = ref_logits(m, ids)
                r = subprocess.run([exe, path, *[x for x in mode if x != "snapshot"], *map(str, ids)], capture_output=True, text=True,
                                   env={**os.environ, **env, **({"TIERLM_SNAPSHOT": "1"} if mode == ["snapshot"] else {})})
                if r.returncode != 0: sys.exit(f"{name}: tierlm-logits failed: {r.stderr.strip()[-3000:]}")
                got = np.array([float(v) for v in r.stdout.split()])
                worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
                top &= bool(got.argmax() == want.argmax())
        for ids in (tests[2], long_ids):                                # three conversations at once
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
        print(f"{name:11s} {label:7s}: worst relative diff {worst:.1e} (limit {tol:.0e}), top token matches: {top}  ->  {'ok' if passed else 'FAIL'}")
    os.remove(path)
sys.exit(0 if ok else 1)
