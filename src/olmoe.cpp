#include "olmoe.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace tierlm {

static constexpr int kGraphSize = 16384;

OlmoeModel::~OlmoeModel() {
    if (sched_) ggml_backend_sched_free(sched_);
    if (kvbuf_) ggml_backend_buffer_free(kvbuf_);
    if (wbuf_) ggml_backend_buffer_free(wbuf_);
    if (kvctx_) ggml_free(kvctx_);
    if (wctx_) ggml_free(wctx_);
    if (main_ && main_ != cpu_) ggml_backend_free(main_);
    if (cpu_) ggml_backend_free(cpu_);
}

std::unique_ptr<OlmoeModel> OlmoeModel::load(const std::string& path, int max_ctx, const std::string& device) {
    std::unique_ptr<OlmoeModel> m(new OlmoeModel());

    // ---- pick the device ----
    m->cpu_ = ggml_backend_cpu_init();
    if (!m->cpu_) throw std::runtime_error("failed to start the CPU backend");
    m->main_ = m->cpu_;
    if (device != "cpu") {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (dev) {
            m->dev_ = dev;
            m->main_ = ggml_backend_dev_init(dev, nullptr);
            if (!m->main_) throw std::runtime_error(std::string("failed to start ") + ggml_backend_dev_name(dev));
            m->device_name_ = std::string(ggml_backend_dev_name(dev)) + " (" + ggml_backend_dev_description(dev) + ")";
        } else if (device == "gpu") {
            throw std::runtime_error("no GPU backend in this build (or no GPU found)");
        }
    }
    if (m->main_ == m->cpu_) {
        m->device_name_ = "CPU";
        m->dev_ = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }

    // ---- read the header: tensor shapes/types only, no data yet ----
    gguf_init_params params{ /*no_alloc=*/true, /*ctx=*/&m->wctx_ };
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

    // ---- allocate every weight on the device, then stream the data in from the file ----
    m->wbuf_ = ggml_backend_alloc_ctx_tensors(m->wctx_, m->main_);
    if (!m->wbuf_) { gguf_free(g); throw std::runtime_error("not enough device memory for the weights"); }
    ggml_backend_buffer_set_usage(m->wbuf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    m->weights_bytes_ = ggml_backend_buffer_get_size(m->wbuf_);

    std::ifstream file(path, std::ios::binary);
    if (!file) { gguf_free(g); throw std::runtime_error("cannot open " + path); }
    const size_t data_off = gguf_get_data_offset(g);
    std::vector<char> chunk;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        ggml_tensor* t = ggml_get_tensor(m->wctx_, gguf_get_tensor_name(g, i));
        const size_t n = ggml_nbytes(t);
        chunk.resize(n);
        file.seekg((std::streamoff)(data_off + gguf_get_tensor_offset(g, i)));
        if (!file.read(chunk.data(), (std::streamsize)n)) { gguf_free(g); throw std::runtime_error("short read on " + path); }
        ggml_backend_tensor_set(t, chunk.data(), 0, n);
    }
    gguf_free(g);

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

    // ---- KV cache on the same device: f32 for now (exact); smaller types come later ----
    const int hd = hp.n_embd / hp.n_head;
    m->max_ctx_ = max_ctx > 0 ? max_ctx : hp.n_ctx;
    ggml_init_params kp{ (size_t)2 * hp.n_layer * ggml_tensor_overhead() + 1024, nullptr, /*no_alloc=*/true };
    m->kvctx_ = ggml_init(kp);
    for (auto& L : m->layers_) {
        L.k_cache = ggml_new_tensor_3d(m->kvctx_, GGML_TYPE_F32, hd, hp.n_head_kv, m->max_ctx_);
        L.v_cache = ggml_new_tensor_3d(m->kvctx_, GGML_TYPE_F32, hd, hp.n_head_kv, m->max_ctx_);
    }
    m->kvbuf_ = ggml_backend_alloc_ctx_tensors(m->kvctx_, m->main_);
    if (!m->kvbuf_) throw std::runtime_error("not enough device memory for the KV cache (try a smaller -c)");
    ggml_backend_buffer_clear(m->kvbuf_, 0);
    m->kv_bytes_ = ggml_backend_buffer_get_size(m->kvbuf_);

    // ---- scheduler: runs each op on the device, falling back to the CPU where needed ----
    ggml_backend_t backends[2] = { m->main_, m->cpu_ };
    const int n_backends = m->main_ == m->cpu_ ? 1 : 2;
    m->sched_ = ggml_backend_sched_new(backends, nullptr, n_backends, kGraphSize, false, false);
    if (!m->sched_) throw std::runtime_error("failed to create the scheduler");
    m->graph_mem_.resize(ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false));
    return m;
}

void OlmoeModel::device_memory(size_t& free, size_t& total) const {
    free = total = 0;
    if (dev_) ggml_backend_dev_memory((ggml_backend_dev_t)dev_, &free, &total);
}

std::vector<float> OlmoeModel::eval(const std::vector<int32_t>& ids, int n_threads, std::vector<int32_t>* routing) {
    const OlmoeHparams& hp = hp_;
    const int n_tok = (int)ids.size();
    if (n_tok == 0) return {};
    const int n_past = n_past_;
    const int n_kv = n_past + n_tok;              // tokens visible to attention after this call
    if (n_kv > max_ctx_) throw std::runtime_error("context full (" + std::to_string(max_ctx_) + " tokens)");
    const int hd = hp.n_embd / hp.n_head;

    // The graph is only a description here; the scheduler allocates its memory on the right device.
    ggml_init_params ip{ graph_mem_.size(), graph_mem_.data(), /*no_alloc=*/true };
    ggml_context* c = ggml_init(ip);

    ggml_tensor* inp = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tok);
    ggml_set_input(inp);
    ggml_tensor* pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tok);
    ggml_set_input(pos);

    ggml_cgraph* gf = ggml_new_graph_custom(c, kGraphSize, false);
    std::vector<ggml_tensor*> orders;                                      // router rankings, read back if asked
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
        ggml_tensor* order = ggml_argsort(c, probs, GGML_SORT_ORDER_DESC);  // [n_expert, n_tok] best first
        ggml_tensor* sel   = ggml_view_2d(c, order, hp.n_expert_used, n_tok, order->nb[1], 0);   // top n_used
        if (routing) { ggml_set_output(order); orders.push_back(order); }
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
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) { ggml_free(c); throw std::runtime_error("failed to allocate compute memory"); }

    std::vector<int32_t> positions(n_tok);
    for (int i = 0; i < n_tok; ++i) positions[i] = n_past + i;
    ggml_backend_tensor_set(inp, ids.data(), 0, sizeof(int32_t) * n_tok);
    ggml_backend_tensor_set(pos, positions.data(), 0, sizeof(int32_t) * n_tok);
    ggml_backend_cpu_set_n_threads(cpu_, n_threads);

    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(c);
        throw std::runtime_error("graph compute failed");
    }
    std::vector<float> out(hp.n_vocab);
    ggml_backend_tensor_get(logits, out.data(), 0, sizeof(float) * hp.n_vocab);
    if (routing) {
        const int L = (int)orders.size(), K = hp.n_expert_used;
        routing->assign((size_t)n_tok * L * K, 0);
        std::vector<int32_t> buf((size_t)hp.n_expert * n_tok);
        for (int l = 0; l < L; ++l) {
            ggml_backend_tensor_get(orders[l], buf.data(), 0, sizeof(int32_t) * buf.size());
            for (int t = 0; t < n_tok; ++t)
                for (int k = 0; k < K; ++k)
                    (*routing)[((size_t)t * L + l) * K + k] = buf[(size_t)t * hp.n_expert + k];
        }
    }
    ggml_free(c);
    n_past_ = n_kv;
    return out;
}

}
