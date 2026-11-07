#include "encoder.h"
#include "backends.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <stdexcept>

namespace tierlm {

bool is_encoder_arch(const std::string& arch) { return arch == "bert" || arch == "nomic-bert"; }

Encoder::~Encoder() {
    if (sched_) ggml_backend_sched_free(sched_);
    if (wbuf_) ggml_backend_buffer_free(wbuf_);
    if (main_ && main_ != cpu_) ggml_backend_free(main_);
    if (cpu_) { forget_cpu_backend(cpu_); ggml_backend_free(cpu_); }
    if (wctx_) ggml_free(wctx_);
    if (gguf_) gguf_free(gguf_);
}

std::unique_ptr<Encoder> Encoder::load(const std::string& path, const std::string& device) {
    std::unique_ptr<Encoder> e(new Encoder());
    gguf_init_params gp{ /*no_alloc=*/true, &e->wctx_ };
    e->gguf_ = gguf_init_from_file(path.c_str(), gp);
    if (!e->gguf_) throw std::runtime_error("failed to read GGUF: " + path);
    gguf_context* g = e->gguf_;
    auto sval = [&](const std::string& k) -> std::string {
        int64_t id = gguf_find_key(g, k.c_str());
        return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING ? gguf_get_val_str(g, id) : "";
    };
    auto num = [&](const std::string& k, double def) -> double {
        int64_t id = gguf_find_key(g, k.c_str());
        if (id < 0) return def;
        switch (gguf_get_kv_type(g, id)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
            case GGUF_TYPE_INT32: return gguf_get_val_i32(g, id);
            case GGUF_TYPE_UINT64: return (double)gguf_get_val_u64(g, id);
            case GGUF_TYPE_INT64: return (double)gguf_get_val_i64(g, id);
            case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(g, id);
            case GGUF_TYPE_UINT16: return gguf_get_val_u16(g, id);
            case GGUF_TYPE_UINT8: return gguf_get_val_u8(g, id);
            default: return def;
        }
    };
    e->arch_ = sval("general.architecture");
    if (!is_encoder_arch(e->arch_)) throw std::runtime_error("'" + e->arch_ + "' is not a supported embedding model");
    const std::string a = e->arch_ + ".";
    e->n_embd_ = (int)num(a + "embedding_length", 0);
    e->n_head_ = (int)num(a + "attention.head_count", 0);
    e->n_layer_ = (int)num(a + "block_count", 0);
    e->n_ctx_ = (int)num(a + "context_length", 512);
    e->eps_ = (float)num(a + "attention.layer_norm_epsilon", 1e-12);
    e->rope_base_ = (float)num(a + "rope.freq_base", 10000.0);
    const int pool = (int)num(a + "pooling_type", 1);
    e->pooling_ = pool == 2 ? Pooling::Cls : pool == 3 ? Pooling::Last : Pooling::Mean;   // none/rank: mean
    if (e->n_embd_ <= 0 || e->n_head_ <= 0 || e->n_layer_ <= 0 || e->n_embd_ % e->n_head_)
        throw std::runtime_error("embedding model with incomplete settings");

    auto T = [&](const std::string& name, bool need) -> ggml_tensor* {
        ggml_tensor* t = ggml_get_tensor(e->wctx_, name.c_str());
        if (!t && need) throw std::runtime_error("embedding model is missing " + name);
        return t;
    };
    e->tok_embd_ = T("token_embd.weight", true);
    e->type_embd_ = T("token_types.weight", false);
    e->pos_embd_ = T("position_embd.weight", false);
    e->embd_norm_ = T("token_embd_norm.weight", true);
    e->embd_norm_b_ = T("token_embd_norm.bias", false);
    e->rope_ = e->arch_ == "nomic-bert";
    if (!e->rope_ && !e->pos_embd_) throw std::runtime_error("BERT model without position embeddings");
    for (int il = 0; il < e->n_layer_; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        Layer L;
        L.qkv = T(b + "attn_qkv.weight", false);   L.qkv_b = T(b + "attn_qkv.bias", false);
        if (!L.qkv) {
            L.q = T(b + "attn_q.weight", true); L.q_b = T(b + "attn_q.bias", false);
            L.k = T(b + "attn_k.weight", true); L.k_b = T(b + "attn_k.bias", false);
            L.v = T(b + "attn_v.weight", true); L.v_b = T(b + "attn_v.bias", false);
        }
        L.o = T(b + "attn_output.weight", true);   L.o_b = T(b + "attn_output.bias", false);
        L.attn_norm = T(b + "attn_output_norm.weight", true); L.attn_norm_b = T(b + "attn_output_norm.bias", false);
        L.up = T(b + "ffn_up.weight", true);       L.up_b = T(b + "ffn_up.bias", false);
        L.gate = T(b + "ffn_gate.weight", false);
        L.down = T(b + "ffn_down.weight", true);   L.down_b = T(b + "ffn_down.bias", false);
        L.out_norm = T(b + "layer_output_norm.weight", true); L.out_norm_b = T(b + "layer_output_norm.bias", false);
        e->layers_.push_back(L);
    }
    for (ggml_tensor* t = ggml_get_first_tensor(e->wctx_); t; t = ggml_get_next_tensor(e->wctx_, t)) e->weight_bytes_ += ggml_nbytes(t);

    // ---- the device: a GPU when the whole model fits with plenty to spare (a chat model may be using most of
    // the VRAM), else the CPU ----
    load_backends();
    e->cpu_ = cpu_backend_init();
    if (!e->cpu_) throw std::runtime_error("failed to start the CPU backend");
    e->main_ = e->cpu_;
    ggml_backend_dev_t dev = nullptr;
    // "auto:<device>": the chat model's GPU (as "CUDA0"); the embedding model goes there when it has room, else on
    // the CPU, never on another, slower chip (such as a laptop's built-in graphics next to its real GPU).
    std::string mode = device;
    ggml_backend_dev_t pref = nullptr;
    if (device.rfind("auto:", 0) == 0) { pref = ggml_backend_dev_by_name(device.substr(5).c_str()); mode = "auto"; }
    const size_t need = e->weight_bytes_ + e->weight_bytes_ / 2 + ((size_t)768 << 20);   // weights + working memory
    auto room = [&](ggml_backend_dev_t d) { size_t free = 0, total = 0; ggml_backend_dev_memory(d, &free, &total); return free; };
    if (mode == "auto" || mode == "gpu") {
        if (pref) dev = room(pref) >= need ? pref : nullptr;
        else {
            bool any_discrete = false;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) any_discrete |= ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
            size_t best = 0;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                const auto t = ggml_backend_dev_type(d);
                if (t != GGML_BACKEND_DEVICE_TYPE_GPU && !(t == GGML_BACKEND_DEVICE_TYPE_IGPU && !any_discrete)) continue;
                const size_t f = room(d);
                if (f >= need && f > best) { dev = d; best = f; }
            }
        }
    } else if (mode != "cpu") {
        dev = ggml_backend_dev_by_name(mode.c_str());
        if (!dev) throw std::runtime_error("unknown device '" + mode + "'");
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) dev = nullptr;
    }
    if (dev) {
        e->main_ = ggml_backend_dev_init(dev, nullptr);
        if (!e->main_) throw std::runtime_error(std::string("failed to start ") + ggml_backend_dev_name(dev));
        e->device_name_ = std::string(ggml_backend_dev_name(dev)) + " (" + ggml_backend_dev_description(dev) + ")";
    } else e->device_name_ = "CPU";

    // ---- the weights, read straight into their buffer ----
    {
        std::lock_guard<std::recursive_mutex> lk(device_mutex());
        e->wbuf_ = ggml_backend_alloc_ctx_tensors(e->wctx_, e->main_);
        if (!e->wbuf_) throw std::runtime_error("not enough memory for the embedding model");
        ggml_backend_buffer_set_usage(e->wbuf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        const size_t base = gguf_get_data_offset(g);
        std::vector<char> buf;
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            ggml_tensor* t = ggml_get_tensor(e->wctx_, gguf_get_tensor_name(g, i));
            if (!t) continue;
            buf.resize(ggml_nbytes(t));
            f.seekg((std::streamoff)(base + gguf_get_tensor_offset(g, i)));
            if (!f.read(buf.data(), (std::streamsize)buf.size())) throw std::runtime_error("embedding model file is cut short");
            ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
        }
    }
    std::vector<ggml_backend_t> backends;
    if (e->main_ != e->cpu_) backends.push_back(e->main_);
    backends.push_back(e->cpu_);
    e->sched_ = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), 8192, false, true);
    if (!e->sched_) throw std::runtime_error("failed to create the scheduler");
    return e;
}

std::vector<float> Encoder::embed(const std::vector<int32_t>& ids_in, int n_threads) {
    if (ids_in.empty()) throw std::runtime_error("nothing to embed");
    std::vector<int32_t> ids(ids_in.begin(), ids_in.begin() + std::min<size_t>(ids_in.size(), (size_t)n_ctx_));
    const int n = (int)ids.size(), E = n_embd_, H = n_head_, hd = E / H;
    const size_t GS = 8192;
    ggml_init_params ip{ ggml_tensor_overhead() * GS + ggml_graph_overhead_custom(GS, false), nullptr, true };
    ggml_context* c = ggml_init(ip);
    ggml_tensor* inp = ggml_new_tensor_1d(c, GGML_TYPE_I32, n); ggml_set_input(inp);
    ggml_tensor* pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n); ggml_set_input(pos);
    auto norm = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {        // LayerNorm with weight and bias
        x = ggml_mul(c, ggml_norm(c, x, eps_), w);
        return b ? ggml_add(c, x, b) : x;
    };
    auto lin = [&](ggml_tensor* w, ggml_tensor* b, ggml_tensor* x) {
        ggml_tensor* y = ggml_mul_mat(c, w, x);
        return b ? ggml_add(c, y, b) : y;
    };
    ggml_tensor* x = ggml_get_rows(c, tok_embd_, inp);                         // [E, n]
    if (type_embd_) x = ggml_add(c, x, ggml_view_1d(c, type_embd_, E, 0));    // every token is sentence A (type 0)
    if (pos_embd_) x = ggml_add(c, x, ggml_get_rows(c, pos_embd_, pos));
    x = norm(x, embd_norm_, embd_norm_b_);
    const float scale = 1.0f / std::sqrt((float)hd);
    for (const Layer& L : layers_) {
        ggml_tensor *q, *k, *v;
        if (L.qkv) {
            ggml_tensor* qkv = lin(L.qkv, L.qkv_b, x);                          // [3E, n]: q, then k, then v
            q = ggml_cont(c, ggml_view_2d(c, qkv, E, n, qkv->nb[1], 0));
            k = ggml_cont(c, ggml_view_2d(c, qkv, E, n, qkv->nb[1], (size_t)E * ggml_element_size(qkv)));
            v = ggml_cont(c, ggml_view_2d(c, qkv, E, n, qkv->nb[1], (size_t)2 * E * ggml_element_size(qkv)));
        } else {
            q = lin(L.q, L.q_b, x); k = lin(L.k, L.k_b, x); v = lin(L.v, L.v_b, x);
        }
        q = ggml_reshape_3d(c, q, hd, H, n);
        k = ggml_reshape_3d(c, k, hd, H, n);
        v = ggml_reshape_3d(c, v, hd, H, n);
        if (rope_) {                                                            // rotate halves (NeoX layout)
            q = ggml_rope_ext(c, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, n_ctx_, rope_base_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(c, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, n_ctx_, rope_base_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        }
        // Attention in both directions: every token sees every other (no mask).
        ggml_tensor* qp = ggml_permute(c, q, 0, 2, 1, 3);                      // [hd, n, H]
        ggml_tensor* kp = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));      // (contiguous: every GPU kernel takes it)
        ggml_tensor* kq = ggml_soft_max_ext(c, ggml_mul_mat(c, kp, qp), nullptr, scale, 0.0f);   // [n_k, n_q, H]
        ggml_tensor* vp = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));        // [n, hd, H]
        ggml_tensor* att = ggml_mul_mat(c, vp, kq);                             // [hd, n_q, H]
        att = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, att, 0, 2, 1, 3)), E, n);
        x = norm(ggml_add(c, x, lin(L.o, L.o_b, att)), L.attn_norm, L.attn_norm_b);
        ggml_tensor* h;
        if (L.gate) h = ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, L.gate, x)), lin(L.up, L.up_b, x));   // nomic: SwiGLU
        else h = ggml_gelu_erf(c, lin(L.up, L.up_b, x));                                          // BERT: exact GELU
        x = norm(ggml_add(c, x, lin(L.down, L.down_b, h)), L.out_norm, L.out_norm_b);
    }
    ggml_set_output(x);
    ggml_cgraph* gf = ggml_new_graph_custom(c, GS, false);
    ggml_build_forward_expand(gf, x);

    std::vector<float> hid((size_t)E * n);
    {
        // One model computes at a time: the chat model and this one take turns between steps.
        std::lock_guard<std::recursive_mutex> lk(device_mutex());
        ggml_backend_sched_reset(sched_);
        if (!ggml_backend_sched_alloc_graph(sched_, gf)) { ggml_free(c); throw std::runtime_error("not enough memory to embed this text"); }
        std::vector<int32_t> p(n);
        for (int i = 0; i < n; ++i) p[i] = i;
        ggml_backend_tensor_set(inp, ids.data(), 0, sizeof(int32_t) * n);
        ggml_backend_tensor_set(pos, p.data(), 0, sizeof(int32_t) * n);
        set_n_threads(cpu_, n_threads);
        if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) { ggml_free(c); throw std::runtime_error("embedding failed"); }
        ggml_backend_tensor_get(x, hid.data(), 0, sizeof(float) * hid.size());
    }
    ggml_free(c);
    std::vector<float> out(E, 0.0f);
    if (pooling_ == Pooling::Cls) std::copy(hid.begin(), hid.begin() + E, out.begin());
    else if (pooling_ == Pooling::Last) std::copy(hid.end() - E, hid.end(), out.begin());
    else {
        for (int t = 0; t < n; ++t) for (int i = 0; i < E; ++i) out[i] += hid[(size_t)t * E + i];
        for (float& v : out) v /= (float)n;
    }
    return out;
}

}
