#include "olmoe.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace tierlm {

OlmoeModel::~OlmoeModel() {
    if (kvctx_) ggml_free(kvctx_);
    if (wctx_) ggml_free(wctx_);
}

std::unique_ptr<OlmoeModel> OlmoeModel::load(const std::string& path, int max_ctx) {
    std::unique_ptr<OlmoeModel> m(new OlmoeModel());
    gguf_init_params params{ /*no_alloc=*/false, /*ctx=*/&m->wctx_ };   // reads all weights into RAM
    gguf_context* g = gguf_init_from_file(path.c_str(), params);
    if (!g) throw std::runtime_error("failed to load GGUF: " + path);

    auto str = [&](const char* k) -> std::string { int64_t id = gguf_find_key(g, k); return id < 0 ? "" : gguf_get_val_str(g, id); };
    auto u32 = [&](const std::string& k, bool required) -> int {
        int64_t id = gguf_find_key(g, k.c_str());
        if (id < 0) { if (required) throw std::runtime_error("missing key " + k); return 0; }
        return (int)gguf_get_val_u32(g, id);
    };
    auto f32 = [&](const std::string& k, float def) -> float {
        int64_t id = gguf_find_key(g, k.c_str());
        return id < 0 ? def : gguf_get_val_f32(g, id);
    };

    const std::string arch = str("general.architecture");
    if (arch != "olmoe") { gguf_free(g); throw std::runtime_error("expected olmoe, got " + arch); }
    OlmoeHparams& hp = m->hp_;
    hp.n_layer       = u32(arch + ".block_count", true);
    hp.n_embd        = u32(arch + ".embedding_length", true);
    hp.n_head        = u32(arch + ".attention.head_count", true);
    hp.n_head_kv     = u32(arch + ".attention.head_count_kv", false);
    hp.n_ff          = u32(arch + ".feed_forward_length", true);
    hp.n_expert      = u32(arch + ".expert_count", true);
    hp.n_expert_used = u32(arch + ".expert_used_count", true);
    hp.n_ctx         = u32(arch + ".context_length", false);
    hp.rope_base     = f32(arch + ".rope.freq_base", 10000.0f);
    hp.rms_eps       = f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
    if (hp.n_head_kv == 0) hp.n_head_kv = hp.n_head;
    if (hp.n_ctx == 0) hp.n_ctx = 4096;
    gguf_free(g);   // tensor data stays in wctx_

    auto T = [&](const std::string& name, bool required = true) -> ggml_tensor* {
        ggml_tensor* t = ggml_get_tensor(m->wctx_, name.c_str());
        if (!t && required) throw std::runtime_error("missing tensor " + name);
        return t;
    };
    m->tok_embd_ = T("token_embd.weight");
    m->out_norm_ = T("output_norm.weight");
    m->output_   = T("output.weight", false);
    if (!m->output_) m->output_ = m->tok_embd_;   // tied embeddings
    hp.n_vocab = (int)m->tok_embd_->ne[1];

    for (int i = 0; i < hp.n_layer; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        Layer L;
        L.attn_norm = T(b + "attn_norm.weight");
        L.wq        = T(b + "attn_q.weight");
        L.wk        = T(b + "attn_k.weight");
        L.wv        = T(b + "attn_v.weight");
        L.wo        = T(b + "attn_output.weight");
        L.q_norm    = T(b + "attn_q_norm.weight");
        L.k_norm    = T(b + "attn_k_norm.weight");
        L.ffn_norm  = T(b + "ffn_norm.weight");
        L.router    = T(b + "ffn_gate_inp.weight");
        L.gate      = T(b + "ffn_gate_exps.weight");
        L.up        = T(b + "ffn_up_exps.weight");
        L.down      = T(b + "ffn_down_exps.weight");
        m->layers_.push_back(L);
    }

    // KV cache: f32 for now (exact); smaller types come later.
    const int hd = hp.n_embd / hp.n_head;
    m->max_ctx_ = max_ctx > 0 ? max_ctx : hp.n_ctx;
    const size_t kv_bytes = (size_t)2 * hp.n_layer * hd * hp.n_head_kv * m->max_ctx_ * sizeof(float);
    ggml_init_params kp{ kv_bytes + (size_t)2 * hp.n_layer * ggml_tensor_overhead() + 4096, nullptr, false };
    m->kvctx_ = ggml_init(kp);
    if (!m->kvctx_) throw std::runtime_error("failed to allocate KV cache");
    for (auto& L : m->layers_) {
        L.k_cache = ggml_new_tensor_3d(m->kvctx_, GGML_TYPE_F32, hd, hp.n_head_kv, m->max_ctx_);
        L.v_cache = ggml_new_tensor_3d(m->kvctx_, GGML_TYPE_F32, hd, hp.n_head_kv, m->max_ctx_);
    }
    return m;
}

std::vector<float> OlmoeModel::eval(const std::vector<int32_t>& ids, int n_threads) {
    const OlmoeHparams& hp = hp_;
    const int n_tok = (int)ids.size();
    if (n_tok == 0) return {};
    const int n_past = n_past_;
    const int n_kv = n_past + n_tok;              // tokens visible to attention after this call
    if (n_kv > max_ctx_) throw std::runtime_error("context full (" + std::to_string(max_ctx_) + " tokens)");
    const int hd = hp.n_embd / hp.n_head;

    // Scratch memory for activations, sized generously for this sequence length.
    const size_t per_layer = (size_t)n_tok * ((size_t)hp.n_embd * 4 * 24 + (size_t)hp.n_expert * 4 * 4 +
                                              (size_t)hp.n_expert_used * hp.n_ff * 4 * 4 +
                                              (size_t)hp.n_head * n_kv * 4 * 4);
    const size_t mem = (size_t)256 * 1024 * 1024 + per_layer * hp.n_layer + (size_t)hp.n_vocab * 4 * 2;
    ggml_init_params ip{ mem, nullptr, false };
    ggml_context* c = ggml_init(ip);
    if (!c) throw std::runtime_error("failed to allocate compute memory");

    ggml_tensor* inp = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tok);
    std::memcpy(inp->data, ids.data(), sizeof(int32_t) * n_tok);
    ggml_tensor* pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tok);
    for (int i = 0; i < n_tok; ++i) ((int32_t*)pos->data)[i] = n_past + i;

    ggml_cgraph* gf = ggml_new_graph_custom(c, 16384, false);
    ggml_tensor* x = ggml_get_rows(c, tok_embd_, inp);                     // [n_embd, n_tok]
    for (const Layer& L : layers_) {
        // ---- attention ----
        ggml_tensor* h = ggml_mul(c, ggml_rms_norm(c, x, hp.rms_eps), L.attn_norm);
        ggml_tensor* q = ggml_mul_mat(c, L.wq, h);
        ggml_tensor* k = ggml_mul_mat(c, L.wk, h);
        ggml_tensor* v = ggml_mul_mat(c, L.wv, h);
        q = ggml_mul(c, ggml_rms_norm(c, q, hp.rms_eps), L.q_norm);          // OLMoE: QK-norm over the full width
        k = ggml_mul(c, ggml_rms_norm(c, k, hp.rms_eps), L.k_norm);
        q = ggml_reshape_3d(c, q, hd, hp.n_head, n_tok);
        k = ggml_reshape_3d(c, k, hd, hp.n_head_kv, n_tok);
        v = ggml_reshape_3d(c, v, hd, hp.n_head_kv, n_tok);
        q = ggml_rope_ext(c, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, hp.n_ctx, hp.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(c, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, hp.n_ctx, hp.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // Write this call's keys/values into the cache at positions [n_past, n_kv). These copies are
        // added to the graph first, so they run before the attention below reads the cache.
        const size_t knb1 = L.k_cache->nb[1], knb2 = L.k_cache->nb[2];
        const size_t vnb1 = L.v_cache->nb[1], vnb2 = L.v_cache->nb[2];
        ggml_build_forward_expand(gf, ggml_cpy(c, k, ggml_view_3d(c, L.k_cache, hd, hp.n_head_kv, n_tok, knb1, knb2, (size_t)n_past * knb2)));
        ggml_build_forward_expand(gf, ggml_cpy(c, v, ggml_view_3d(c, L.v_cache, hd, hp.n_head_kv, n_tok, vnb1, vnb2, (size_t)n_past * vnb2)));
        ggml_tensor* kall = ggml_view_3d(c, L.k_cache, hd, hp.n_head_kv, n_kv, knb1, knb2, 0);   // all tokens so far
        ggml_tensor* vall = ggml_view_3d(c, L.v_cache, hd, hp.n_head_kv, n_kv, vnb1, vnb2, 0);

        ggml_tensor* Q  = ggml_permute(c, q, 0, 2, 1, 3);                    // [hd, n_tok, n_head]
        ggml_tensor* K  = ggml_permute(c, kall, 0, 2, 1, 3);                 // [hd, n_kv, n_head_kv]
        ggml_tensor* kq = ggml_mul_mat(c, K, Q);                             // [n_kv, n_tok, n_head]
        kq = ggml_scale(c, kq, 1.0f / std::sqrt((float)hd));
        kq = ggml_diag_mask_inf(c, kq, n_past);                              // causal: no peeking ahead
        kq = ggml_soft_max(c, kq);
        ggml_tensor* V = ggml_cont(c, ggml_permute(c, vall, 1, 2, 0, 3));   // [n_kv, hd, n_head_kv]
        ggml_tensor* o = ggml_mul_mat(c, V, kq);                             // [hd, n_tok, n_head]
        o = ggml_cont(c, ggml_permute(c, o, 0, 2, 1, 3));                    // [hd, n_head, n_tok]
        o = ggml_reshape_2d(c, o, hp.n_embd, n_tok);
        x = ggml_add(c, x, ggml_mul_mat(c, L.wo, o));

        // ---- mixture of experts ----
        h = ggml_mul(c, ggml_rms_norm(c, x, hp.rms_eps), L.ffn_norm);
        ggml_tensor* probs = ggml_soft_max(c, ggml_mul_mat(c, L.router, h)); // [n_expert, n_tok]
        ggml_tensor* sel   = ggml_argsort_top_k(c, probs, hp.n_expert_used); // [n_used, n_tok] expert ids
        ggml_tensor* w     = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, hp.n_expert, n_tok), sel); // [1, n_used, n_tok]
        ggml_tensor* h3    = ggml_reshape_3d(c, h, hp.n_embd, 1, n_tok);
        ggml_tensor* up    = ggml_mul_mat_id(c, L.up, h3, sel);              // [n_ff, n_used, n_tok]
        ggml_tensor* gate  = ggml_mul_mat_id(c, L.gate, h3, sel);
        ggml_tensor* act   = ggml_mul(c, ggml_silu(c, gate), up);
        ggml_tensor* down  = ggml_mul_mat_id(c, L.down, act, sel);           // [n_embd, n_used, n_tok]
        down = ggml_mul(c, down, w);                                         // weight each expert's output
        ggml_tensor* moe = nullptr;
        for (int e = 0; e < hp.n_expert_used; ++e) {
            ggml_tensor* s = ggml_view_2d(c, down, hp.n_embd, n_tok, down->nb[2], e * down->nb[1]);
            moe = moe ? ggml_add(c, moe, s) : s;
        }
        x = ggml_add(c, x, moe);
    }

    x = ggml_mul(c, ggml_rms_norm(c, x, hp.rms_eps), out_norm_);
    ggml_tensor* last   = ggml_view_2d(c, x, hp.n_embd, 1, x->nb[1], (size_t)(n_tok - 1) * x->nb[1]);
    ggml_tensor* logits = ggml_mul_mat(c, output_, last);                    // [n_vocab, 1]

    ggml_build_forward_expand(gf, logits);
    if (ggml_graph_compute_with_ctx(c, gf, n_threads) != GGML_STATUS_SUCCESS) {
        ggml_free(c);
        throw std::runtime_error("graph compute failed");
    }
    std::vector<float> out((const float*)logits->data, (const float*)logits->data + hp.n_vocab);
    ggml_free(c);
    n_past_ = n_kv;
    return out;
}

}
