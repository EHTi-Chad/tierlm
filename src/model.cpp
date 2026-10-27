#include "model.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace tierlm {

static const char* kArchs[] = { "llama", "qwen2", "qwen3", "qwen3moe", "olmoe" };
bool arch_supported(const std::string& arch) {
    for (const char* a : kArchs) if (arch == a) return true;
    return false;
}
std::vector<std::string> supported_archs() { return std::vector<std::string>(std::begin(kArchs), std::end(kArchs)); }

static constexpr int kGraphSize = 16384;

Model::~Model() {
    if (sched_) ggml_backend_sched_free(sched_);
    if (kvbuf_) ggml_backend_buffer_free(kvbuf_);
    if (wbuf_) ggml_backend_buffer_free(wbuf_);
    if (kvctx_) ggml_free(kvctx_);
    if (wctx_) ggml_free(wctx_);
    if (main_ && main_ != cpu_) ggml_backend_free(main_);
    if (cpu_) ggml_backend_free(cpu_);
}

std::unique_ptr<Model> Model::load(const std::string& path, const LoadOptions& opt) {
    std::unique_ptr<Model> m(new Model());
    const std::string& device = opt.device;
    const int max_ctx = opt.max_ctx;
    m->batch_ = opt.batch > 0 ? std::min(opt.batch, 2048) : 256;

    // ---- pick the device ----
    m->cpu_ = ggml_backend_cpu_init();
    if (!m->cpu_) throw std::runtime_error("failed to start the CPU backend");
    m->main_ = m->cpu_;
    if (device != "cpu") {
        ggml_backend_dev_t dev = nullptr;
        if (device == "auto" || device == "gpu") {
            // Prefer a dedicated GPU (the one with the most memory), then an integrated one.
            // The same NVIDIA card can appear twice (CUDA and Vulkan); CUDA wins when the memory is about equal.
            size_t best = 0;
            auto is_cuda = [](ggml_backend_dev_t d) {
                ggml_backend_reg_t r = ggml_backend_dev_backend_reg(d);
                return r && std::string(ggml_backend_reg_name(r)) == "CUDA";
            };
            const size_t slack = (size_t)512 << 20;
            for (enum ggml_backend_dev_type want : { GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU }) {
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    if (ggml_backend_dev_type(d) != want) continue;
                    size_t free = 0, total = 0;
                    ggml_backend_dev_memory(d, &free, &total);
                    const bool better = !dev || total > best + slack ||
                                        (total + slack >= best && is_cuda(d) && !is_cuda(dev));
                    if (better) { dev = d; best = total; }
                }
                if (dev) break;
            }
        } else {
            dev = ggml_backend_dev_by_name(device.c_str());   // a specific device, e.g. "Vulkan0" or "CUDA0"
            if (!dev) throw std::runtime_error("unknown device '" + device + "' (tierlm-devices lists them)");
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) dev = nullptr;
        }
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
    if (!arch_supported(arch)) { gguf_free(g); throw std::runtime_error("the " + arch + " architecture is not supported yet"); }
    Hparams& hp = m->hp_;
    hp.arch          = arch;
    hp.n_layer       = u32(arch + ".block_count", true);
    hp.n_embd        = u32(arch + ".embedding_length", true);
    hp.n_head        = u32(arch + ".attention.head_count", true);
    hp.n_head_kv     = u32(arch + ".attention.head_count_kv", false);
    hp.head_dim      = u32(arch + ".attention.key_length", false);
    hp.n_rot         = u32(arch + ".rope.dimension_count", false);
    hp.n_ff          = u32(arch + ".feed_forward_length", false);
    hp.n_ff_exp      = u32(arch + ".expert_feed_forward_length", false);
    hp.n_expert      = u32(arch + ".expert_count", false);
    hp.n_expert_used = u32(arch + ".expert_used_count", false);
    hp.n_ctx         = u32(arch + ".context_length", false);
    hp.rope_base     = f32(arch + ".rope.freq_base", 10000.0f);
    hp.rms_eps       = f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
    if (hp.n_head_kv == 0) hp.n_head_kv = hp.n_head;
    if (hp.head_dim == 0) hp.head_dim = hp.n_embd / hp.n_head;
    if (hp.n_rot == 0) hp.n_rot = hp.head_dim;
    if (hp.n_ctx == 0) hp.n_ctx = 4096;
    if (hp.n_expert > 0 && hp.n_ff_exp == 0) hp.n_ff_exp = hp.n_ff;
    if (hp.n_expert > 0 && hp.n_expert_used == 0) { gguf_free(g); throw std::runtime_error("missing key " + arch + ".expert_used_count"); }
    {   // RoPE scaling: "linear" stretches positions; Llama 3.1-style factors come as a rope_freqs tensor instead
        const std::string type = str((arch + ".rope.scaling.type").c_str());
        const float factor = f32(arch + ".rope.scaling.factor", 1.0f);
        if (type == "linear" && factor > 0) hp.rope_scale = 1.0f / factor;
        else if (!type.empty() && type != "none" && factor > 1.0f)
            std::fprintf(stderr, "warning: rope scaling '%s' is not supported yet; contexts beyond the original length will degrade\n", type.c_str());
    }
    // The variations
    hp.rope_neox     = arch != "llama";
    hp.qk_norm       = arch == "olmoe" ? Hparams::kFull : (arch == "qwen3" || arch == "qwen3moe") ? Hparams::kPerHead : Hparams::kNone;
    hp.qkv_bias      = arch == "qwen2";
    hp.expert_renorm = arch == "qwen3moe" || arch == "llama";   // Mixtral and Qwen3-MoE renormalise; OLMoE doesn't
    if (arch == "qwen3moe") {
        int64_t id = gguf_find_key(g, (arch + ".expert_weights_norm").c_str());
        if (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_BOOL) hp.expert_renorm = gguf_get_val_bool(g, id);
    }
    m->n_expert_used_file_ = hp.n_expert_used;
    if (opt.experts_used > 0 && hp.n_expert > 0) hp.n_expert_used = std::min(opt.experts_used, hp.n_expert);

    // ---- allocate every weight on the device, then stream the data in from the file ----
    m->wbuf_ = ggml_backend_alloc_ctx_tensors(m->wctx_, m->main_);
    if (!m->wbuf_) { gguf_free(g); throw std::runtime_error("not enough device memory for the weights"); }
    ggml_backend_buffer_set_usage(m->wbuf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    m->weights_bytes_ = ggml_backend_buffer_get_size(m->wbuf_);

    std::ifstream file(std::filesystem::u8path(path), std::ios::binary);   // UTF-8 path, also on Windows
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
    m->tok_embd_   = T("token_embd.weight");
    m->out_norm_   = T("output_norm.weight");
    m->output_     = T("output.weight", false);
    m->rope_freqs_ = T("rope_freqs.weight", false);   // Llama 3.1+ long-context frequency factors
    if (!m->output_) m->output_ = m->tok_embd_;      // tied embeddings
    hp.n_vocab = (int)m->tok_embd_->ne[1];

    for (int i = 0; i < hp.n_layer; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        Layer L;
        L.attn_norm = T(b + "attn_norm.weight");
        L.wq        = T(b + "attn_q.weight");
        L.wk        = T(b + "attn_k.weight");
        L.wv        = T(b + "attn_v.weight");
        L.wo        = T(b + "attn_output.weight");
        L.bq        = T(b + "attn_q.bias", hp.qkv_bias);
        L.bk        = T(b + "attn_k.bias", hp.qkv_bias);
        L.bv        = T(b + "attn_v.bias", hp.qkv_bias);
        if (hp.qk_norm != Hparams::kNone) {
            L.q_norm = T(b + "attn_q_norm.weight");
            L.k_norm = T(b + "attn_k_norm.weight");
        }
        L.ffn_norm  = T(b + "ffn_norm.weight");
        if (hp.is_moe()) {
            L.router = T(b + "ffn_gate_inp.weight");
            L.gate_x = T(b + "ffn_gate_exps.weight", false);
            if (!L.gate_x) throw std::runtime_error("this file stores experts as separate tensors (an old Mixtral GGUF); re-download a newer conversion");
            L.up_x   = T(b + "ffn_up_exps.weight");
            L.down_x = T(b + "ffn_down_exps.weight");
        } else {
            L.gate = T(b + "ffn_gate.weight");
            L.up   = T(b + "ffn_up.weight");
            L.down = T(b + "ffn_down.weight");
        }
        m->layers_.push_back(L);
    }

    // ---- KV cache on the same device, in f16: half the size and memory traffic of f32 ----
    const int hd = hp.head_dim;
    m->max_ctx_ = max_ctx > 0 ? max_ctx : hp.n_ctx;
    // TIERLM_KV_F32=1 keeps it in f32 (exact; used by the tests to check the attention math).
    m->kv_f32_ = opt.kv_f32 >= 0 ? opt.kv_f32 == 1 : std::getenv("TIERLM_KV_F32") != nullptr;
    const ggml_type kv_type = m->kv_f32_ ? GGML_TYPE_F32 : GGML_TYPE_F16;
    ggml_init_params kp{ (size_t)2 * hp.n_layer * ggml_tensor_overhead() + 1024, nullptr, /*no_alloc=*/true };
    m->kvctx_ = ggml_init(kp);
    for (auto& L : m->layers_) {
        L.k_cache = ggml_new_tensor_3d(m->kvctx_, kv_type, hd, hp.n_head_kv, m->max_ctx_);
        L.v_cache = ggml_new_tensor_3d(m->kvctx_, kv_type, hd, hp.n_head_kv, m->max_ctx_);
    }
    m->kvbuf_ = ggml_backend_alloc_ctx_tensors(m->kvctx_, m->main_);
    if (!m->kvbuf_) throw std::runtime_error("not enough device memory for the KV cache (try a smaller -c)");
    ggml_backend_buffer_clear(m->kvbuf_, 0);
    m->kv_bytes_ = ggml_backend_buffer_get_size(m->kvbuf_);

    // Fused flash attention reads the cache once, with no copies. TIERLM_NO_FA=1 uses the plain version.
    m->flash_attn_ = opt.flash_attn >= 0 ? opt.flash_attn == 1 : std::getenv("TIERLM_NO_FA") == nullptr;

    // ---- scheduler: runs each op on the device, falling back to the CPU where needed ----
    ggml_backend_t backends[2] = { m->main_, m->cpu_ };
    const int n_backends = m->main_ == m->cpu_ ? 1 : 2;
    m->sched_ = ggml_backend_sched_new(backends, nullptr, n_backends, kGraphSize, false, false);
    if (!m->sched_) throw std::runtime_error("failed to create the scheduler");
    m->graph_mem_.resize(ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false));

    // Reserve compute memory once, for the largest graph we will ever build (a full batch at the end
    // of the context), so it is not re-planned on every token.
    const int n_res = std::min(m->batch_, m->max_ctx_);
    Graph worst = m->build(n_res, m->max_ctx_ - n_res, true);
    bool ok = ggml_backend_sched_reserve(m->sched_, worst.gf);
    ggml_free(worst.ctx);
    if (!ok) throw std::runtime_error("failed to reserve compute memory");
    return m;
}

bool Model::integrated_gpu() const {
    return dev_ && ggml_backend_dev_type((ggml_backend_dev_t)dev_) == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

void Model::device_memory(size_t& free, size_t& total) const {
    free = total = 0;
    if (dev_) ggml_backend_dev_memory((ggml_backend_dev_t)dev_, &free, &total);
}

// Builds the compute graph for n_tok new tokens placed after n_past cached ones.
// Only a description: the scheduler allocates its memory on the right device later.
Model::Graph Model::build(int n_tok, int n_past, bool want_routing) {
    const Hparams& hp = hp_;
    const int n_kv = n_past + n_tok;
    const int hd = hp.head_dim;

    Graph g;
    ggml_init_params ip{ graph_mem_.size(), graph_mem_.data(), /*no_alloc=*/true };
    ggml_context* c = g.ctx = ggml_init(ip);
    g.gf = ggml_new_graph_custom(c, kGraphSize, false);

    g.inp = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tok);
    ggml_set_input(g.inp);
    g.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tok);
    ggml_set_input(g.pos);
    if (flash_attn_) {                                                     // causal mask: 0 = visible, -inf = future
        g.mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv, n_tok);
        ggml_set_input(g.mask);
    }
    const float scale = 1.0f / std::sqrt((float)hd);

    ggml_tensor* x = ggml_get_rows(c, tok_embd_, g.inp);                   // [n_embd, n_tok]
    for (const Layer& L : layers_) {
        // ---- attention ----
        ggml_tensor* h = ggml_mul(c, ggml_rms_norm(c, x, hp.rms_eps), L.attn_norm);
        ggml_tensor* q = ggml_mul_mat(c, L.wq, h);                            // [n_head*hd, n_tok]
        ggml_tensor* k = ggml_mul_mat(c, L.wk, h);
        ggml_tensor* v = ggml_mul_mat(c, L.wv, h);
        if (L.bq) { q = ggml_add(c, q, L.bq); k = ggml_add(c, k, L.bk); v = ggml_add(c, v, L.bv); }
        if (hp.qk_norm == Hparams::kFull) {                                  // OLMoE: over the full width
            q = ggml_mul(c, ggml_rms_norm(c, q, hp.rms_eps), L.q_norm);
            k = ggml_mul(c, ggml_rms_norm(c, k, hp.rms_eps), L.k_norm);
        }
        q = ggml_reshape_3d(c, q, hd, hp.n_head, n_tok);
        k = ggml_reshape_3d(c, k, hd, hp.n_head_kv, n_tok);
        v = ggml_reshape_3d(c, v, hd, hp.n_head_kv, n_tok);
        if (hp.qk_norm == Hparams::kPerHead) {                               // Qwen3: per head
            q = ggml_mul(c, ggml_rms_norm(c, q, hp.rms_eps), L.q_norm);
            k = ggml_mul(c, ggml_rms_norm(c, k, hp.rms_eps), L.k_norm);
        }
        const int mode = hp.rope_neox ? GGML_ROPE_TYPE_NEOX : 0;
        q = ggml_rope_ext(c, q, g.pos, rope_freqs_, hp.n_rot, mode, hp.n_ctx, hp.rope_base, hp.rope_scale, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(c, k, g.pos, rope_freqs_, hp.n_rot, mode, hp.n_ctx, hp.rope_base, hp.rope_scale, 0.0f, 1.0f, 32.0f, 1.0f);

        // Write this batch's keys/values into the cache at positions [n_past, n_kv). These copies are
        // added to the graph first, so they run before the attention below reads the cache.
        const size_t knb1 = L.k_cache->nb[1], knb2 = L.k_cache->nb[2];
        const size_t vnb1 = L.v_cache->nb[1], vnb2 = L.v_cache->nb[2];
        ggml_build_forward_expand(g.gf, ggml_cpy(c, k, ggml_view_3d(c, L.k_cache, hd, hp.n_head_kv, n_tok, knb1, knb2, (size_t)n_past * knb2)));
        ggml_build_forward_expand(g.gf, ggml_cpy(c, v, ggml_view_3d(c, L.v_cache, hd, hp.n_head_kv, n_tok, vnb1, vnb2, (size_t)n_past * vnb2)));
        ggml_tensor* kall = ggml_view_3d(c, L.k_cache, hd, hp.n_head_kv, n_kv, knb1, knb2, 0);   // all tokens so far
        ggml_tensor* vall = ggml_view_3d(c, L.v_cache, hd, hp.n_head_kv, n_kv, vnb1, vnb2, 0);

        ggml_tensor* Q = ggml_permute(c, q, 0, 2, 1, 3);                     // [hd, n_tok, n_head]
        ggml_tensor* K = ggml_permute(c, kall, 0, 2, 1, 3);                  // [hd, n_kv, n_head_kv]
        ggml_tensor* o;
        if (flash_attn_) {
            ggml_tensor* V = ggml_permute(c, vall, 0, 2, 1, 3);              // [hd, n_kv, n_head_kv], no copy
            o = ggml_flash_attn_ext(c, Q, K, V, g.mask, scale, 0.0f, 0.0f);  // [hd, n_head, n_tok]
        } else {
            ggml_tensor* kq = ggml_mul_mat(c, K, Q);                         // [n_kv, n_tok, n_head]
            kq = ggml_scale(c, kq, scale);
            kq = ggml_diag_mask_inf(c, kq, n_past);                          // causal: no peeking ahead
            kq = ggml_soft_max(c, kq);
            ggml_tensor* V = ggml_cont(c, ggml_permute(c, vall, 1, 2, 0, 3));   // [n_kv, hd, n_head_kv]
            o = ggml_mul_mat(c, V, kq);                                      // [hd, n_tok, n_head]
            o = ggml_cont(c, ggml_permute(c, o, 0, 2, 1, 3));                // [hd, n_head, n_tok]
        }
        o = ggml_reshape_2d(c, o, hd * hp.n_head, n_tok);
        x = ggml_add(c, x, ggml_mul_mat(c, L.wo, o));

        h = ggml_mul(c, ggml_rms_norm(c, x, hp.rms_eps), L.ffn_norm);
        if (!hp.is_moe()) {
            // ---- dense SwiGLU feed-forward ----
            ggml_tensor* act = ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, L.gate, h)), ggml_mul_mat(c, L.up, h));
            x = ggml_add(c, x, ggml_mul_mat(c, L.down, act));
            continue;
        }
        // ---- mixture of experts ----
        ggml_tensor* probs = ggml_soft_max(c, ggml_mul_mat(c, L.router, h)); // [n_expert, n_tok]
        ggml_tensor* order = ggml_argsort(c, probs, GGML_SORT_ORDER_DESC);  // [n_expert, n_tok] best first
        ggml_tensor* sel   = ggml_view_2d(c, order, hp.n_expert_used, n_tok, order->nb[1], 0);   // top n_used
        if (want_routing) { ggml_set_output(order); g.orders.push_back(order); }
        ggml_tensor* w     = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, hp.n_expert, n_tok), sel); // [1, n_used, n_tok]
        if (hp.expert_renorm) {                                             // chosen weights sum to 1
            ggml_tensor* w2 = ggml_reshape_2d(c, w, hp.n_expert_used, n_tok);
            w2 = ggml_div(c, w2, ggml_sum_rows(c, w2));
            w = ggml_reshape_3d(c, w2, 1, hp.n_expert_used, n_tok);
        }
        ggml_tensor* h3    = ggml_reshape_3d(c, h, hp.n_embd, 1, n_tok);
        ggml_tensor* up    = ggml_mul_mat_id(c, L.up_x, h3, sel);            // [n_ff_exp, n_used, n_tok]
        ggml_tensor* gate  = ggml_mul_mat_id(c, L.gate_x, h3, sel);
        ggml_tensor* act   = ggml_mul(c, ggml_silu(c, gate), up);
        ggml_tensor* down  = ggml_mul_mat_id(c, L.down_x, act, sel);         // [n_embd, n_used, n_tok]
        down = ggml_mul(c, down, w);                                         // weight each expert's output
        ggml_tensor* moe = nullptr;
        for (int e = 0; e < hp.n_expert_used; ++e) {
            ggml_tensor* s = ggml_view_2d(c, down, hp.n_embd, n_tok, down->nb[2], e * down->nb[1]);
            moe = moe ? ggml_add(c, moe, s) : s;
        }
        x = ggml_add(c, x, moe);
    }

    x = ggml_mul(c, ggml_rms_norm(c, x, hp.rms_eps), out_norm_);
    ggml_tensor* last = ggml_view_2d(c, x, hp.n_embd, 1, x->nb[1], (size_t)(n_tok - 1) * x->nb[1]);
    g.logits = ggml_mul_mat(c, output_, last);                             // [n_vocab, 1]
    ggml_set_output(g.logits);
    ggml_build_forward_expand(g.gf, g.logits);
    return g;
}

// Long prompts are processed in batches of batch_ tokens, which bounds the memory a single graph needs.
std::vector<float> Model::eval(const std::vector<int32_t>& ids, int n_threads, std::vector<int32_t>* routing) {
    if (ids.empty()) return {};
    if (n_past_ + (int)ids.size() > max_ctx_) throw std::runtime_error("context full (" + std::to_string(max_ctx_) + " tokens)");
    if (routing) routing->clear();
    std::vector<float> lg;
    std::vector<int32_t> part;
    for (size_t off = 0; off < ids.size(); off += (size_t)batch_) {
        const size_t n = std::min<size_t>((size_t)batch_, ids.size() - off);
        lg = eval_batch(ids.data() + off, (int)n, n_threads, routing ? &part : nullptr);
        if (routing) routing->insert(routing->end(), part.begin(), part.end());
    }
    return lg;
}

std::vector<float> Model::eval_batch(const int32_t* ids, int n_tok, int n_threads, std::vector<int32_t>* routing) {
    const Hparams& hp = hp_;
    const int n_past = n_past_, n_kv = n_past + n_tok;
    Graph g = build(n_tok, n_past, routing != nullptr);

    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, g.gf)) { ggml_free(g.ctx); throw std::runtime_error("failed to allocate compute memory"); }

    std::vector<int32_t> positions(n_tok);
    for (int i = 0; i < n_tok; ++i) positions[i] = n_past + i;
    ggml_backend_tensor_set(g.inp, ids, 0, sizeof(int32_t) * n_tok);
    ggml_backend_tensor_set(g.pos, positions.data(), 0, sizeof(int32_t) * n_tok);
    if (g.mask) {
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask((size_t)n_kv * n_tok);
        for (int t = 0; t < n_tok; ++t)
            for (int j = 0; j < n_kv; ++j) mask[(size_t)t * n_kv + j] = j <= n_past + t ? zero : ninf;
        ggml_backend_tensor_set(g.mask, mask.data(), 0, sizeof(ggml_fp16_t) * mask.size());
    }
    ggml_backend_cpu_set_n_threads(cpu_, n_threads);

    if (ggml_backend_sched_graph_compute(sched_, g.gf) != GGML_STATUS_SUCCESS) {
        ggml_free(g.ctx);
        throw std::runtime_error("graph compute failed");
    }
    std::vector<float> out(hp.n_vocab);
    ggml_backend_tensor_get(g.logits, out.data(), 0, sizeof(float) * hp.n_vocab);
    if (routing) {
        const int L = (int)g.orders.size(), K = hp.n_expert_used;
        routing->assign((size_t)n_tok * L * K, 0);
        std::vector<int32_t> buf((size_t)hp.n_expert * n_tok);
        for (int l = 0; l < L; ++l) {
            ggml_backend_tensor_get(g.orders[l], buf.data(), 0, sizeof(int32_t) * buf.size());
            for (int t = 0; t < n_tok; ++t)
                for (int k = 0; k < K; ++k)
                    (*routing)[((size_t)t * L + l) * K + k] = buf[(size_t)t * hp.n_expert + k];
        }
    }
    ggml_free(g.ctx);
    n_past_ = n_kv;
    return out;
}

}
