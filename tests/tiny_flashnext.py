"""Qwen3.8 Flash-Next (GGUF arch "qwen4exp") against Hugging Face's own code.

Builds tiny random Flash-Next models with transformers' Qwen4ExpForCausalLM (hyper-connections, the per-layer n-gram
embedding with its dilated convolution, Gated DeltaNet and gated attention layers, experts with a gated shared
expert), writes each as a GGUF laid out as the real Unsloth files are (tensor names and settings checked against
Qwen3.8-Flash-Next-Q4_K_M: norm weights stored as 1 + w, ssm_a as -exp(A_log), linear-attention value heads in
tiled order), runs tierlm-logits on it and compares the logits with transformers' for the same tokens.

The first cases stay within the sparse attention's token budget (plain attention); the qsa ones go far past it.

usage: python tests/tiny_flashnext.py build/tierlm-logits
"""
import os, subprocess, sys, tempfile
import numpy as np
import torch
import gguf
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

V, EOS = 300, 1
BASE = dict(vocab_size=V, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
            moe_intermediate_size=24, shared_expert_intermediate_size=24, num_experts=16, num_experts_per_tok=4,
            hc_count=4, hc_lowrank=16, ple_layer_ids=[2], ple_embed_dim=64, ngram_size=3, heads_per_ngram=2,
            ngram_vocab_size_base=1000, make_ngram_vocab_size_divisible_by=8, split_ngram_parts=1,
            indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=512, indexer_compress_ratio=4,
            eos_token_id=EOS, max_position_embeddings=1024, rms_norm_eps=1e-6,
            output_gate_type="sigmoid",   # as the released model's config (the file doesn't record it; HF defaults to silu)
            rope_parameters=dict(rope_type="default", rope_theta=10000.0, partial_rotary_factor=0.5, mrope_section=[2, 1, 1]),
            attn_implementation="eager")
CASES = {
    "flashnext":   dict(),
    "two-ple":     dict(ple_layer_ids=[1, 3], num_hidden_layers=4),        # PLE on two layers, the first one at layer 0
    "eight-layer": dict(num_hidden_layers=8, ple_layer_ids=[2]),
    # QSA past its budget: each query attends only to the blocks its indexer ranks highest (tiny budgets, so the
    # test contexts are far past them)
    "qsa-8":       dict(indexer_budget=8),
    "qsa-16":      dict(indexer_budget=16, num_hidden_layers=8),
}

def build(over, seed):
    torch.manual_seed(seed)
    cfg = Qwen4ExpTextConfig(**{**BASE, **over})
    m = Qwen4ExpForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("A_log"): p.copy_(torch.log(torch.empty_like(p).uniform_(0.5, 4)))
            elif n.endswith("dt_bias"): p.copy_(0.5 + 0.1 * torch.randn_like(p))
            elif "linear_attn.norm" in n: p.copy_(1 + 0.1 * torch.randn_like(p))          # gated norm: plain weight
            elif "norm" in n: p.copy_(0.1 * torch.randn_like(p))                          # (1 + w) norms: around 0
            else: p.copy_(0.2 * torch.randn_like(p))
    return cfg, m

def write(cfg, m):
    sd = {k: v.detach().float().numpy() for k, v in m.state_dict().items()}
    E, HC, HK, HV, S = cfg.hidden_size, cfg.hc_count, cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim
    R = HV // HK
    def tiled(x, axis, head):          # value heads: grouped by key head -> tiled (as Qwen3.5's converter)
        x = np.moveaxis(x, axis, 0); sh = x.shape
        x = x.reshape(HK, R, head, *sh[1:]).transpose(1, 0, 2, *range(3, len(sh) + 2)).reshape(sh)
        return np.ascontiguousarray(np.moveaxis(x, 0, axis))
    T = {"token_embd.weight": sd["model.embed_tokens.weight"], "output.weight": sd["lm_head.weight"],
         "output_hc_norm.weight": 1 + sd["model.hyper_connection_mixer.hc_norm.weight"],
         "output_hc_down.weight": sd["model.hyper_connection_mixer.input_mix_weight_down.weight"],
         "output_hc_up.weight": sd["model.hyper_connection_mixer.input_mix_weight_up.weight"]}
    ple_layers = []
    for i in range(cfg.num_hidden_layers):
        p, b = f"model.layers.{i}.", f"blk.{i}."
        for blk, nm in (("attn", "attn_hyper_connection"), ("ffn", "mlp_hyper_connection")):
            T[b + f"hc_{blk}_norm.weight"] = 1 + sd[p + nm + ".hc_norm.weight"]
            T[b + f"hc_{blk}_down.weight"] = sd[p + nm + ".input_mix_weight_down.weight"]
            T[b + f"hc_{blk}_up.weight"] = sd[p + nm + ".input_mix_weight_up.weight"]
            T[b + f"hc_{blk}_inject.weight"] = sd[p + nm + ".block_inject_weight.weight"]
        if cfg.layer_types[i] == "linear_attention":
            a = p + "linear_attn."
            qk = 2 * HK * S
            qkv, conv = sd[a + "in_proj_qkv.weight"], sd[a + "conv1d.weight"][:, 0, :]
            T.update({b + "attn_qkv.weight": np.concatenate([qkv[:qk], tiled(qkv[qk:], 0, S)]),
                      b + "attn_gate.weight": tiled(sd[a + "in_proj_z.weight"], 0, S),
                      b + "ssm_beta.weight": tiled(sd[a + "in_proj_b.weight"], 0, 1), b + "ssm_alpha.weight": tiled(sd[a + "in_proj_a.weight"], 0, 1),
                      b + "ssm_conv1d.weight": np.concatenate([conv[:qk], tiled(conv[qk:], 0, S)]),
                      b + "ssm_dt.bias": tiled(sd[a + "dt_bias"], 0, 1), b + "ssm_a": tiled(-np.exp(sd[a + "A_log"]), 0, 1),
                      b + "ssm_norm.weight": sd[a + "norm.weight"], b + "ssm_out.weight": tiled(sd[a + "out_proj.weight"], 1, S)})
        else:
            a = p + "self_attn."
            T.update({b + "attn_q.weight": sd[a + "q_proj.weight"], b + "attn_k.weight": sd[a + "k_proj.weight"],
                      b + "attn_v.weight": sd[a + "v_proj.weight"], b + "attn_output.weight": sd[a + "o_proj.weight"],
                      b + "attn_q_norm.weight": 1 + sd[a + "q_norm.weight"], b + "attn_k_norm.weight": 1 + sd[a + "k_norm.weight"]})
            qk = sd[a + "indexer.index_qk_proj.weight"]; nq = cfg.indexer_n_heads * cfg.indexer_head_dim
            T.update({b + "indexer.q_proj.weight": qk[:nq], b + "indexer.k_proj.weight": qk[nq:],
                      b + "indexer.q_norm.weight": 1 + sd[a + "indexer.q_layernorm.weight"], b + "indexer.k_norm.weight": 1 + sd[a + "indexer.k_layernorm.weight"]})
        mp = p + "mlp."
        gu = sd[mp + "experts.gate_up_proj"]; F = gu.shape[1] // 2
        T.update({b + "ffn_gate_inp.weight": sd[mp + "gate.weight"], b + "ffn_gate_exps.weight": np.ascontiguousarray(gu[:, :F, :]),
                  b + "ffn_up_exps.weight": np.ascontiguousarray(gu[:, F:, :]), b + "ffn_down_exps.weight": sd[mp + "experts.down_proj"],
                  b + "ffn_gate_shexp.weight": sd[mp + "shared_expert.gate_proj.weight"], b + "ffn_up_shexp.weight": sd[mp + "shared_expert.up_proj.weight"],
                  b + "ffn_down_shexp.weight": sd[mp + "shared_expert.down_proj.weight"], b + "ffn_gate_inp_shexp.weight": sd[mp + "shared_expert_gate.weight"].reshape(-1)})
        if p + "ple.key_proj.weight" in sd:
            ple_layers.append(i)
            q = p + "ple."
            T.update({b + "ple_key.weight": sd[q + "key_proj.weight"], b + "ple_value.weight": sd[q + "value_proj.weight"],
                      b + "ple_norm_key.weight": 1 + sd[q + "norm_key.weight"], b + "ple_norm_query.weight": 1 + sd[q + "norm_query.weight"],
                      b + "ple_norm_conv.weight": 1 + sd[q + "norm_conv.weight"], b + "ple_conv1d.weight": sd[q + "conv1d.weight"][:, 0, :]})
    # one table holding every PLE layer's rows (the real file has one PLE layer; with more, they follow each other)
    emb = [m.model.layers[i].ple.ple_embedding for i in ple_layers]
    T["per_layer_token_embd.weight"] = np.concatenate([e.ngram_embedding.weight.detach().float().numpy() for e in emb])
    path = os.path.join(tempfile.mkdtemp(), "tiny-flashnext.gguf")
    w = gguf.GGUFWriter(path, "qwen4exp"); A = "qwen4exp"
    w.add_block_count(cfg.num_hidden_layers); w.add_context_length(cfg.max_position_embeddings); w.add_embedding_length(E)
    w.add_head_count(cfg.num_attention_heads); w.add_head_count_kv(cfg.num_key_value_heads)
    w.add_array(f"{A}.rope.dimension_sections", list(cfg.rope_parameters["mrope_section"]) + [0])
    w.add_rope_freq_base(cfg.rope_parameters["rope_theta"]); w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    w.add_expert_count(cfg.num_experts); w.add_expert_used_count(cfg.num_experts_per_tok)
    w.add_key_length(cfg.head_dim); w.add_value_length(cfg.head_dim)
    w.add_expert_feed_forward_length(cfg.moe_intermediate_size); w.add_uint32(f"{A}.expert_shared_feed_forward_length", cfg.shared_expert_intermediate_size)
    w.add_uint32(f"{A}.ssm.conv_kernel", cfg.linear_conv_kernel_dim); w.add_uint32(f"{A}.ssm.state_size", S)
    w.add_uint32(f"{A}.ssm.group_count", HK); w.add_uint32(f"{A}.ssm.time_step_rank", HV); w.add_uint32(f"{A}.ssm.inner_size", HV * S)
    w.add_uint32(f"{A}.full_attention_interval", 4)
    w.add_rope_dimension_count(int(cfg.head_dim * cfg.rope_parameters["partial_rotary_factor"]))
    w.add_uint32(f"{A}.hyper_connection.count", HC); w.add_uint32(f"{A}.hyper_connection.low_rank", cfg.hc_lowrank)
    w.add_uint32(f"{A}.attention.indexer.head_count", cfg.indexer_n_heads); w.add_uint32(f"{A}.attention.indexer.key_length", cfg.indexer_head_dim)
    w.add_uint32(f"{A}.attention.indexer.top_k", cfg.indexer_budget)
    w.add_array(f"{A}.attention.compress_ratios", [cfg.indexer_compress_ratio if t != "linear_attention" else 0 for t in cfg.layer_types])
    w.add_array(f"{A}.ple.layers", ple_layers)
    w.add_uint32(f"{A}.ple.ngram_size", cfg.ngram_size); w.add_uint32(f"{A}.ple.heads_per_ngram", cfg.heads_per_ngram)
    w.add_uint32(f"{A}.ple.conv_kernel", cfg.ple_conv_kernel_size); w.add_uint32(f"{A}.ple.eos_token_id", EOS)
    w.add_uint32(f"{A}.embedding_length_per_layer_input", cfg.ple_embed_dim // ((cfg.ngram_size - 1) * cfg.heads_per_ngram))
    # the hash numbers, one PLE layer's at a time in the real file; here the first one's (all layers share the layout
    # check below) and the per-layer offsets into the joined table
    e0 = emb[0]
    u64 = lambda k, v: w.add_key_value(k, [int(x) for x in v], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.UINT64)
    u64(f"{A}.ple.layer_multipliers", e0.layer_multipliers)          # (as the real file: 64-bit numbers)
    u64(f"{A}.ple.head_offsets", e0.ngram_heads_offsets)
    u64(f"{A}.ple.head_vocab_sizes", e0.ngram_heads_vocab_sizes)
    for name, t in T.items(): w.add_tensor(name, np.ascontiguousarray(t).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return path, len(ple_layers)

def ref_logits(m, ids):
    with torch.no_grad():
        return m(torch.tensor([ids])).logits[0, -1].double().numpy()

exe = sys.argv[1]
long_ids = [int(x) for x in np.random.default_rng(3).integers(2, V, 200)]
long_ids[37] = long_ids[120] = EOS                                  # end-of-text tokens cut the n-grams
tests = [[5], [1, 2, 3], [7, 42, 99, 0, 13, 13, 64, 8, 21, 55, 117, EOS, 4, 4, 9], long_ids]
ok = True
for ci, (name, over) in enumerate(CASES.items()):
    cfg, m = build(over, 30 + ci)
    path, n_ple = write(cfg, m)
    if n_ple != 1:
        print(f"{name:12s}: skipped (tierlm reads one PLE layer's hash settings, as the real file has)"); os.remove(path); continue
    for label, env, tol in (("exact", {"TIERLM_KV_F32": "1", "TIERLM_DEVICE": "cpu"}, 1e-4),
                            ("normal", {}, 6e-2)):
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
        for ids in (tests[2], long_ids):                    # three conversations at once
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
        # Speculative decoding (scripted guesses wrong at different depths, and prompt lookup): greedy output must not
        # change, so undoing the guesses that weren't kept restores every running state (DeltaNet's, the PLE
        # convolution's, the n-gram history, QSA's keys). The prompt is past the QSA budget in the qsa cases.
        prompt = [str(x) for x in long_ids[:40]]
        def gen(mode, extra=None):
            r = subprocess.run([exe, path, mode, "50", *prompt], capture_output=True, text=True, env={**os.environ, **env, **(extra or {})})
            if r.returncode != 0: sys.exit(f"{name}: tierlm-logits {mode} failed: {r.stderr.strip()[-800:]}")
            return r.stdout.split()
        plain = gen("--gen")
        checks = [gen("--gen-spec", {"TIERLM_DRAFT_N": n, "TIERLM_DRAFT_SCRIPT": ",".join(plain), "TIERLM_DRAFT_CORRUPT": k})
                  for n, k in (("1", "1"), ("3", "3"), ("3", "5"), ("8", "2"), ("8", "7"))]
        checks.append(gen("--gen-spec", {"TIERLM_LOOKUP": "8"}))
        same = all(c == plain for c in checks)
        ok &= same
        print(f"{name:12s} {label:7s}: greedy output with speculation {'identical' if same else 'DIFFERENT'}  ->  {'ok' if same else 'FAIL'}")
    os.remove(path)
sys.exit(0 if ok else 1)
