"""Checks tests/tiny_deepseek4.py's NumPy reference against Hugging Face's own DeepSeek V4 code (transformers >= 5.23, torch):
the same tiny model's weights loaded into DeepseekV4ForCausalLM, logits compared for the whole sequence and one token
at a time with its cache. (The tiny model's SwiGLU clamps are set equal here: Hugging Face has one for all layers.)"""
import sys, numpy as np, torch
sys.argv = ['x']
src = open(__import__('os').path.join(__import__('os').path.dirname(__file__), 'tiny_deepseek4.py')).read().split('if __name__')[0]
g = {}
exec(src, g)
L, RATIOS = g['L'], g['RATIOS']
g['CLAMP'][:] = [0.8] * L; g['CLAMP_S'][:] = [0.8] * L
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
lt = {0: "sliding_attention", 4: "compressed_sparse_attention", 16: "heavily_compressed_attention"}
cfg = DeepseekV4Config(vocab_size=g['V'], hidden_size=g['E'], moe_intermediate_size=g['FE'], num_hidden_layers=L,
    num_attention_heads=g['H'], num_key_value_heads=1, head_dim=g['HD'], q_lora_rank=g['QL'],
    num_experts_per_tok=g['NU'], n_routed_experts=g['NE'], n_shared_experts=1, routed_scaling_factor=g['WSCALE'],
    max_position_embeddings=g['CTX'], rope_theta=g['THETA'], layer_types=[lt[r] for r in RATIOS],
    compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 16}, compress_rope_theta=g['CTHETA'],
    hc_mult=g['HC'], hc_sinkhorn_iters=g['SINK_IT'], hc_eps=g['HCEPS'], mlp_layer_types=["hash_moe"] * g['HASH'] + ["moe"] * (L - g['HASH']),
    swiglu_limit=0.8, sliding_window=g['W'], o_groups=g['OG'], o_lora_rank=g['OL'], index_n_heads=g['IH'], index_head_dim=g['IHD'],
    index_topk=g['TOPK'], rms_norm_eps=g['EPS'], partial_rotary_factor=g['ROT'] / g['HD'],
    rope_parameters={"rope_type": "yarn", "factor": g['YF'], "original_max_position_embeddings": g['YORIG'],
                     "beta_fast": g['YBF'], "beta_slow": g['YBS']}, num_nextn_predict_layers=0)
cfg._attn_implementation = "eager"
cfg._experts_implementation = "eager"
m = DeepseekV4ForCausalLM(cfg).double().eval()
P = g['P']
sd = {}
t = lambda a: torch.tensor(np.asarray(a, dtype=np.float64))
sd['model.embed_tokens.weight'] = t(P['embed']); sd['model.norm.weight'] = t(P['norm']); sd['lm_head.weight'] = t(P['head'])
sd['model.hc_head.hc_fn'] = t(P['hh_fn']); sd['model.hc_head.hc_base'] = t(P['hh_base']); sd['model.hc_head.hc_scale'] = t(P['hh_scale'])
for i in range(L):
    p, b = P[i], f'model.layers.{i}.'
    a = b + 'self_attn.'
    sd.update({b + 'input_layernorm.weight': t(p['attn_norm']), b + 'post_attention_layernorm.weight': t(p['ffn_norm']),
               a + 'q_a_proj.weight': t(p['q_a']), a + 'q_a_norm.weight': t(p['q_a_norm']), a + 'q_b_proj.weight': t(p['q_b']),
               a + 'kv_proj.weight': t(p['kv']), a + 'kv_norm.weight': t(p['kv_norm']), a + 'o_a_proj.weight': t(p['o_a']),
               a + 'o_b_proj.weight': t(p['o_b']), a + 'sinks': t(p['sinks'])})
    for site in ('attn', 'ffn'):
        sd[b + f'{site}_hc.fn'] = t(p[site + '_hc_fn']); sd[b + f'{site}_hc.base'] = t(p[site + '_hc_base']); sd[b + f'{site}_hc.scale'] = t(p[site + '_hc_scale'])
    mm = b + 'mlp.'
    sd[mm + 'gate.weight'] = t(p['router'])
    if i < g['HASH']: sd[mm + 'gate.tid2eid'] = torch.tensor(p['tid2eid'].astype(np.int64))
    else: sd[mm + 'gate.e_score_correction_bias'] = t(p['probs_b'])
    sd[mm + 'experts.gate_up_proj'] = t(np.concatenate([p['eg'], p['eu']], 1))
    sd[mm + 'experts.down_proj'] = t(p['ed'])
    sd[mm + 'shared_experts.gate_proj.weight'] = t(p['sg']); sd[mm + 'shared_experts.up_proj.weight'] = t(p['su']); sd[mm + 'shared_experts.down_proj.weight'] = t(p['sd'])
    if RATIOS[i]:
        c = a + 'compressor.'
        sd.update({c + 'kv_proj.weight': t(p['c_kv']), c + 'gate_proj.weight': t(p['c_gate']), c + 'position_bias': t(p['c_ape']), c + 'kv_norm.weight': t(p['c_norm'])})
        if RATIOS[i] == 4:
            ix = c + 'indexer.'
            sd.update({ix + 'q_b_proj.weight': t(p['i_q_b']), ix + 'scorer.weights_proj.weight': t(p['i_proj']),
                       ix + 'kv_proj.weight': t(p['ic_kv']), ix + 'gate_proj.weight': t(p['ic_gate']), ix + 'position_bias': t(p['ic_ape']), ix + 'kv_norm.weight': t(p['ic_norm'])})
missing, unexpected = m.load_state_dict(sd, strict=False)
print('missing', [k for k in missing if 'rotary' not in k][:20]); print('unexpected', unexpected[:20])
for n in (5, 23, 61):
    ids = np.random.default_rng(n).integers(0, g['V'], n)
    with torch.no_grad():
        out = m(torch.tensor(ids[None]), use_cache=False).logits[0, -1].numpy()
    ref = g['reference'](ids)
    print(n, 'tokens: rel diff', float(np.abs(out - ref).max() / np.abs(ref).max()))
    # with the cache, one token at a time
    with torch.no_grad():
        o = m(torch.tensor(ids[None, :1]), use_cache=True); pkv = o.past_key_values
        for k in range(1, n):
            o = m(torch.tensor(ids[None, k:k+1]), past_key_values=pkv, use_cache=True); pkv = o.past_key_values
    print(n, 'tokens stepwise: rel diff', float(np.abs(o.logits[0, -1].numpy() - ref).max() / np.abs(ref).max()))
