"""Gemma 3 (GGUF arch "gemma3", text) against Hugging Face's own code.

Builds tiny random Gemma 3 text models with transformers' Gemma3ForCausalLM (sliding-window layers with their own
RoPE base, full layers with linear RoPE scaling, q/k norms, norms after attention and after the feed-forward block,
GELU, the embedding scaled by sqrt(width), tied output), writes each as a GGUF laid out as llama.cpp's converter writes
Gemma 3 (norm weights stored as 1 + w), runs tierlm-logits on it and compares the logits with transformers' for the
same tokens. The window is small, so the longer prompts go well past it.

usage: python tests/tiny_gemma3.py build/tierlm-logits
"""
import os, subprocess, sys, tempfile
import numpy as np
import torch
import gguf
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

V = 300
BASE = dict(vocab_size=V, hidden_size=64, intermediate_size=96, num_hidden_layers=7, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16, sliding_window=8, max_position_embeddings=1024,
            rms_norm_eps=1e-6, attn_implementation="eager",
            rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                             "full_attention": {"rope_type": "linear", "rope_theta": 1000000.0, "factor": 8.0}})
CASES = {
    "gemma3":       dict(),
    "twelve-layer": dict(num_hidden_layers=12, sliding_window=16),
    "softcap":      dict(final_logit_softcapping=30.0),              # (Gemma 2's final softcap, read if a file has it)
}

def build(over, seed):
    torch.manual_seed(seed)
    cfg = Gemma3TextConfig(**{**BASE, **over})
    m = Gemma3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n: p.copy_(0.1 * torch.randn_like(p))      # (1 + w) norms: around 0
            else: p.copy_(0.2 * torch.randn_like(p))
    return cfg, m

def write(cfg, m):
    sd = {k: v.detach().float().numpy() for k, v in m.state_dict().items()}
    T = {"token_embd.weight": sd["model.embed_tokens.weight"], "output_norm.weight": 1 + sd["model.norm.weight"]}
    for i in range(cfg.num_hidden_layers):
        p, b = f"model.layers.{i}.", f"blk.{i}."
        a, f = p + "self_attn.", p + "mlp."
        T.update({b + "attn_norm.weight": 1 + sd[p + "input_layernorm.weight"],
                  b + "attn_q.weight": sd[a + "q_proj.weight"], b + "attn_k.weight": sd[a + "k_proj.weight"],
                  b + "attn_v.weight": sd[a + "v_proj.weight"], b + "attn_output.weight": sd[a + "o_proj.weight"],
                  b + "attn_q_norm.weight": 1 + sd[a + "q_norm.weight"], b + "attn_k_norm.weight": 1 + sd[a + "k_norm.weight"],
                  b + "post_attention_norm.weight": 1 + sd[p + "post_attention_layernorm.weight"],
                  b + "ffn_norm.weight": 1 + sd[p + "pre_feedforward_layernorm.weight"],
                  b + "post_ffw_norm.weight": 1 + sd[p + "post_feedforward_layernorm.weight"],
                  b + "ffn_gate.weight": sd[f + "gate_proj.weight"], b + "ffn_up.weight": sd[f + "up_proj.weight"],
                  b + "ffn_down.weight": sd[f + "down_proj.weight"]})
    path = os.path.join(tempfile.mkdtemp(), "tiny-gemma3.gguf")
    w = gguf.GGUFWriter(path, "gemma3"); A = "gemma3"
    w.add_block_count(cfg.num_hidden_layers); w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(cfg.hidden_size); w.add_feed_forward_length(cfg.intermediate_size)
    w.add_head_count(cfg.num_attention_heads); w.add_head_count_kv(cfg.num_key_value_heads)
    w.add_key_length(cfg.head_dim); w.add_value_length(cfg.head_dim)
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps); w.add_sliding_window(cfg.sliding_window)
    full = cfg.rope_parameters["full_attention"]
    w.add_rope_freq_base(full["rope_theta"])
    if full.get("rope_type") == "linear":
        w.add_string(f"{A}.rope.scaling.type", "linear"); w.add_float32(f"{A}.rope.scaling.factor", full["factor"])
    if cfg.final_logit_softcapping: w.add_float32(f"{A}.final_logit_softcapping", cfg.final_logit_softcapping)
    for name, t in T.items(): w.add_tensor(name, np.ascontiguousarray(t).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return path

def ref_logits(m, ids):
    with torch.no_grad():
        return m(torch.tensor([ids])).logits[0, -1].double().numpy()

# ggml's CPU computes GELU from a table of 16-bit values (its input rounded to 16 bits, the result stored in 16 bits;
# 0 below -10, x above 10): the exact pass's reference does the same. A value a hair from a rounding step can still
# land on the other side of it in one and not the other, and random layers amplify that: 3e-4, not 1e-4, there.
class GeluTable(torch.nn.Module):
    def forward(self, x):
        g = torch.nn.functional.gelu(x.half().float(), approximate="tanh").half().float()
        return torch.where(x <= -10, torch.zeros_like(x), torch.where(x >= 10, x, g))
gelu_table = GeluTable()
def set_act(m, fn):
    for layer in m.model.layers: layer.mlp.act_fn = fn

exe = sys.argv[1]
long_ids = [int(x) for x in np.random.default_rng(3).integers(2, V, 300)]   # > 256: batched prompts
tests = [[5], [1, 2, 3], [7, 42, 99, 0, 13, 13, 64, 8, 21, 55, 117, 4, 4, 9], long_ids]
ok = True
for ci, (name, over) in enumerate(CASES.items()):
    cfg, m = build(over, 40 + ci)
    path = write(cfg, m)
    exact_act = m.model.layers[0].mlp.act_fn
    for label, env, tol in (("exact", {"TIERLM_KV_F32": "1", "TIERLM_DEVICE": "cpu"}, 3e-4), ("normal", {}, 6e-2)):
        set_act(m, gelu_table if label == "exact" else exact_act)
        worst, top = 0.0, True
        for mode in ([], ["--step"], ["snapshot"]):
            for ids in tests:
                want = ref_logits(m, ids)
                r = subprocess.run([exe, path, *[x for x in mode if x != "snapshot"], *map(str, ids)], capture_output=True, text=True,
                                   env={**os.environ, **env, **({"TIERLM_SNAPSHOT": "1"} if mode == ["snapshot"] else {})})
                if r.returncode != 0: sys.exit(f"{name}: tierlm-logits failed: {r.stderr.strip()[-800:]}")
                got = np.array([float(v) for v in r.stdout.split()])
                worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
                top &= bool(got.argmax() == want.argmax())
        for ids in (tests[2], long_ids):                                # three conversations at once
            r = subprocess.run([exe, path, "--parallel", "3", *map(str, ids)], capture_output=True, text=True, env={**os.environ, **env})
            if r.returncode != 0: sys.exit(f"{name}: tierlm-logits --parallel failed: {r.stderr.strip()[-800:]}")
            got = np.array([float(v) for v in r.stdout.split()]).reshape(3, -1)
            n = len(ids)
            for sq in range(3):
                want = ref_logits(m, [ids[(i + 3 * sq) % n] for i in range(max(1, n - sq))])
                worst = max(worst, float(np.abs(got[sq] - want).max() / np.abs(want).max()))
                top &= bool(got[sq].argmax() == want.argmax())
        passed = worst < tol and top
        ok &= passed
        print(f"{name:12s} {label:7s}: worst relative diff {worst:.1e} (limit {tol:.0e}), top token matches: {top}  ->  {'ok' if passed else 'FAIL'}")
    os.remove(path)
sys.exit(0 if ok else 1)
