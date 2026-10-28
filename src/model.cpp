#include "model.h"
#include "gguf_split.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "backends.h"
#include "gguf.h"
#include <algorithm>
#include <set>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace tierlm {

static const char* kArchs[] = { "llama", "qwen2", "qwen3", "qwen3moe", "olmoe", "qwen35", "qwen35moe", "qwen3next" };
bool arch_supported(const std::string& arch) {
    for (const char* a : kArchs) if (arch == a) return true;
    return false;
}
std::vector<std::string> supported_archs() { return std::vector<std::string>(std::begin(kArchs), std::end(kArchs)); }

static constexpr int kGraphSize = 16384;

Model::~Model() {
    release_pending();
    if (sched_) ggml_backend_sched_free(sched_);
    for (auto* b : kvbufs_) if (b) ggml_backend_buffer_free(b);
    for (auto* b : wbufs_) if (b) ggml_backend_buffer_free(b);
    if (kvctx_) ggml_free(kvctx_);
    if (wctx_) ggml_free(wctx_);
    for (auto* b : gpus_) if (b && b != cpu_) ggml_backend_free(b);
    if (cpu_) ggml_backend_free(cpu_);
}

// A separate MTP draft file for this model (Unsloth publishes the draft block as "mtp-<model>.gguf"): a GGUF in the
// same folder whose name starts with "mtp-", for the same architecture and width, holding only the draft block.
std::string mtp_companion(const std::string& model_path) {
    namespace fs = std::filesystem;
    auto header = [](const std::string& p, std::string& arch, uint32_t& embd, uint32_t& nextn, uint32_t& blocks, bool& has_layer0) {
        gguf_init_params ip{ true, nullptr };
        gguf_context* g = gguf_init_from_file(p.c_str(), ip);
        if (!g) return false;
        auto s = [&](const std::string& k) { int64_t id = gguf_find_key(g, k.c_str()); return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING ? std::string(gguf_get_val_str(g, id)) : std::string(); };
        auto u = [&](const std::string& k) { int64_t id = gguf_find_key(g, k.c_str()); return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_UINT32 ? gguf_get_val_u32(g, id) : 0u; };
        arch = s("general.architecture");
        embd = u(arch + ".embedding_length"); nextn = u(arch + ".nextn_predict_layers"); blocks = u(arch + ".block_count");
        has_layer0 = gguf_find_tensor(g, "blk.0.attn_norm.weight") >= 0;
        gguf_free(g);
        return true;
    };
    std::string arch; uint32_t embd = 0, nextn = 0, blocks = 0; bool l0 = false;
    if (!header(model_path, arch, embd, nextn, blocks, l0)) return {};
    const uint32_t n_layer = blocks - nextn;
    std::error_code ec;
    const fs::path dir = fs::u8path(model_path).parent_path();
    for (const auto& e : fs::directory_iterator(dir.empty() ? fs::path(".") : dir, ec)) {
        const std::string name = e.path().filename().u8string();
        if (name.rfind("mtp-", 0) != 0 || e.path().extension() != ".gguf" || fs::equivalent(e.path(), fs::u8path(model_path), ec)) continue;
        std::string a2; uint32_t e2 = 0, n2 = 0, b2 = 0; bool l02 = true;
        if (!header(e.path().u8string(), a2, e2, n2, b2, l02)) continue;
        if (a2 == arch && e2 == embd && n2 > 0 && b2 - n2 == n_layer && !l02) return e.path().u8string();
    }
    return {};
}

std::unique_ptr<Model> Model::load(const std::string& path, const LoadOptions& opt) {
    std::unique_ptr<Model> m(new Model());
    const std::string& device = opt.device;
    const int max_ctx = opt.max_ctx;
    m->batch_ = opt.batch > 0 ? std::min(opt.batch, 2048) : 256;

    // ---- pick the device ----
    load_backends();
    m->cpu_ = cpu_backend_init();
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
            m->gpus_.push_back(m->main_);
            m->device_name_ = std::string(ggml_backend_dev_name(dev)) + " (" + ggml_backend_dev_description(dev) + ")";
            // More GPUs of the same kind (the same engine: CUDA, Vulkan...) become further VRAM tiers. Only the
            // same engine, so one card that shows up under both CUDA and Vulkan isn't counted twice.
            if (opt.multi_gpu && (device == "auto" || device == "gpu")) {
                auto family = [](ggml_backend_dev_t d) {
                    ggml_backend_reg_t r = ggml_backend_dev_backend_reg(d);
                    std::string n = r ? ggml_backend_reg_name(r) : "";
                    return n.substr(0, n.find('['));             // RPC registries are named RPC[host:port]
                };
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    if (d == dev || ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU || family(d) != family(dev)) continue;
                    ggml_backend* b = ggml_backend_dev_init(d, nullptr);
                    if (!b) { std::fprintf(stderr, "note: could not start %s; not using it\n", ggml_backend_dev_name(d)); continue; }
                    m->gpus_.push_back(b);
                    m->device_name_ += std::string(" + ") + ggml_backend_dev_name(d) + " (" + ggml_backend_dev_description(d) + ")";
                }
            }
        } else if (device == "gpu") {
            throw std::runtime_error("no GPU backend in this build (or no GPU found)");
        }
    }
    if (m->main_ == m->cpu_) {
        m->device_name_ = "CPU";
        m->dev_ = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }

    // ---- read the headers: tensor shapes/types only, no data yet ----
    // A model may be split over several files; part 1 has the settings, every part has some tensors.
    const SplitInfo split = gguf_split_info(path);
    if (!split.missing.empty())
        throw std::runtime_error("this model has " + std::to_string(split.count) + " parts, but " + split.missing + " is missing");
    struct Source { int part; size_t offset; ggml_tensor* t; };
    std::vector<Source> sources;
    gguf_context* g = nullptr;                                     // part 1: the model's settings
    {
        std::vector<gguf_context*> gs;
        std::vector<ggml_context*> cs;
        auto cleanup = [&] { for (auto* x : gs) if (x && x != g) gguf_free(x); for (auto* c : cs) if (c) ggml_free(c); };
        size_t n_total = 0;
        for (const auto& p : split.paths) {
            ggml_context* c = nullptr;
            gguf_init_params params{ /*no_alloc=*/true, /*ctx=*/&c };
            gguf_context* gi = gguf_init_from_file(p.c_str(), params);
            if (!gi) { cleanup(); if (g) gguf_free(g); throw std::runtime_error("failed to load GGUF: " + p); }
            if (!g) g = gi;
            gs.push_back(gi); cs.push_back(c);
            n_total += (size_t)gguf_get_n_tensors(gi);
        }
        ggml_init_params wp{ (n_total + 256) * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };   // + room for an MTP file
        m->wctx_ = ggml_init(wp);
        for (size_t pi = 0; pi < gs.size(); ++pi) {
            const size_t data_off = gguf_get_data_offset(gs[pi]);
            for (int64_t k = 0; k < gguf_get_n_tensors(gs[pi]); ++k) {
                const char* name = gguf_get_tensor_name(gs[pi], k);
                if (ggml_get_tensor(m->wctx_, name)) { cleanup(); gguf_free(g); throw std::runtime_error(std::string("tensor ") + name + " appears in two parts"); }
                ggml_tensor* t = ggml_dup_tensor(m->wctx_, ggml_get_tensor(cs[pi], name));
                ggml_set_name(t, name);
                sources.push_back({(int)pi, data_off + gguf_get_tensor_offset(gs[pi], k), t});
            }
        }
        cleanup();
    }

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
    hp.expert_renorm = arch == "qwen3moe" || arch == "qwen35moe" || arch == "qwen3next" || arch == "llama";   // OLMoE doesn't renormalise
    hp.expert_scale  = f32(arch + ".expert_weights_scale", 0.0f);
    if (arch == "qwen3moe" || arch == "qwen35moe" || arch == "qwen3next") {
        int64_t id = gguf_find_key(g, (arch + ".expert_weights_norm").c_str());
        if (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_BOOL) hp.expert_renorm = gguf_get_val_bool(g, id);
    }
    if (arch == "qwen35" || arch == "qwen35moe" || arch == "qwen3next") {
        hp.hybrid = true;
        // Qwen3-Next (Qwen3-Coder-Next) files keep the linear-attention value heads grouped by key head, as
        // Hugging Face stores them; Qwen3.5's converter reorders them so ggml can broadcast the key heads.
        hp.v_grouped = arch == "qwen3next";
        hp.qk_norm = Hparams::kPerHead;
        // Multi-token-prediction (MTP) draft blocks are stored after the real layers. They only run for
        // speculative decoding (LoadOptions::mtp), and then only the first one.
        hp.n_nextn   = u32(arch + ".nextn_predict_layers", false);
        hp.n_layer  -= hp.n_nextn;
        hp.ssm_conv    = u32(arch + ".ssm.conv_kernel", true);
        hp.ssm_state   = u32(arch + ".ssm.state_size", true);
        hp.ssm_k_heads = u32(arch + ".ssm.group_count", true);
        hp.ssm_v_heads = u32(arch + ".ssm.time_step_rank", true);
        hp.recurrent.assign(hp.n_layer, false);
        const int64_t rid = gguf_find_key(g, (arch + ".attention.recurrent_layers").c_str());
        if (rid >= 0 && gguf_get_kv_type(g, rid) == GGUF_TYPE_ARRAY) {
            const auto t = gguf_get_arr_type(g, rid);
            const void* d = gguf_get_arr_data(g, rid);
            for (int i = 0; i < hp.n_layer && i < (int)gguf_get_arr_n(g, rid); ++i)
                hp.recurrent[i] = t == GGUF_TYPE_INT32 || t == GGUF_TYPE_UINT32 ? ((const int32_t*)d)[i] != 0 : ((const int8_t*)d)[i] != 0;
        } else {
            int interval = u32(arch + ".full_attention_interval", false);
            if (interval <= 0) interval = 4;
            for (int i = 0; i < hp.n_layer; ++i) hp.recurrent[i] = (i + 1) % interval != 0;
        }
        if (hp.ssm_v_heads % hp.ssm_k_heads != 0) { gguf_free(g); throw std::runtime_error("unexpected linear attention head counts"); }
    }
    m->n_expert_used_file_ = hp.n_expert_used;
    // Speculation with a separate MTP draft file: add its draft-block tensors as one more part.
    std::vector<std::string> paths = split.paths;
    if (opt.mtp && !ggml_get_tensor(m->wctx_, ("blk." + std::to_string(hp.n_layer) + ".nextn.eh_proj.weight").c_str())) {
        const std::string comp = mtp_companion(path);
        if (!comp.empty()) {
            ggml_context* c = nullptr;
            gguf_init_params ip{ true, &c };
            gguf_context* gc = gguf_init_from_file(comp.c_str(), ip);
            if (gc) {
                const std::string prefix = "blk." + std::to_string(hp.n_layer) + ".";
                const size_t data_off = gguf_get_data_offset(gc);
                for (int64_t k = 0; k < gguf_get_n_tensors(gc); ++k) {
                    const char* name = gguf_get_tensor_name(gc, k);
                    if (std::string(name).rfind(prefix, 0) != 0 || ggml_get_tensor(m->wctx_, name)) continue;
                    ggml_tensor* t = ggml_dup_tensor(m->wctx_, ggml_get_tensor(c, name));
                    ggml_set_name(t, name);
                    sources.push_back({(int)paths.size(), data_off + gguf_get_tensor_offset(gc, k), t});
                }
                const int64_t id = gguf_find_key(gc, (arch + ".nextn_predict_layers").c_str());
                if (id >= 0) hp.n_nextn = (int)gguf_get_val_u32(gc, id);
                paths.push_back(comp);
                gguf_free(gc);
                if (c) ggml_free(c);
                std::fprintf(stderr, "speculative decoding: draft block from %s\n", comp.c_str());
            }
        }
    }
    // Layers to load: the model's, plus the MTP draft block (stored as layer n_layer) when speculating.
    const bool want_mtp = opt.mtp && hp.n_nextn > 0 &&
                          ggml_get_tensor(m->wctx_, ("blk." + std::to_string(hp.n_layer) + ".nextn.eh_proj.weight").c_str());
    const int nl_all = hp.n_layer + (want_mtp ? 1 : 0);
    if (opt.experts_used > 0 && hp.n_expert > 0) hp.n_expert_used = std::min(opt.experts_used, hp.n_expert);

    // ---- tier loader: decide where each weight lives, VRAM first, then system RAM ----
    // Priority for VRAM: small always-used tensors, then attention, router and output, then the KV cache,
    // then dense feed-forward layers, then expert layers. What doesn't fit stays in RAM, where the CPU
    // computes with it (the scheduler moves activations between the two).
    m->max_ctx_ = max_ctx > 0 ? max_ctx : hp.n_ctx;
    // KV cache precision: f16 by default; f32 is exact (the tests use it to check the attention math, via
    // TIERLM_KV_F32=1); q8_0 halves the memory again with practically no quality loss, q4_0 quarters it.
    // Quantized caches are read by flash attention, so they turn it on, and need a head size that is a whole
    // number of 32-value blocks.
    {
        std::string want = opt.kv_type;
        if (want.empty()) {
            if (const char* e = std::getenv("TIERLM_KV_TYPE")) want = e;
            else want = (opt.kv_f32 >= 0 ? opt.kv_f32 == 1 : std::getenv("TIERLM_KV_F32") != nullptr) ? "f32" : "f16";
        }
        m->kv_t_ = want == "f32" ? GGML_TYPE_F32 : want == "q8_0" ? GGML_TYPE_Q8_0 : want == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
        if (ggml_is_quantized((ggml_type)m->kv_t_) && hp.head_dim % 32 != 0) {
            std::fprintf(stderr, "note: this model's head size (%d) can't use a %s cache; using f16\n", hp.head_dim, want.c_str());
            m->kv_t_ = GGML_TYPE_F16;
        }
        m->kv_f32_ = m->kv_t_ == GGML_TYPE_F32;
    }
    // Conversation memory per layer: a KV cache for attention layers, a small fixed-size state for DeltaNet layers.
    auto layer_mem = [&](int il) -> size_t {
        if (hp.is_recurrent(il))
            return ((size_t)(hp.ssm_conv - 1) * hp.ssm_conv_dim() + (size_t)hp.ssm_state * hp.ssm_state * hp.ssm_v_heads) * 4;
        return (size_t)2 * ggml_row_size((ggml_type)m->kv_t_, (int64_t)hp.head_dim * hp.n_head_kv) * (size_t)m->max_ctx_;
    };
    size_t kv_need = 0;
    for (int il = 0; il < nl_all; ++il) kv_need += layer_mem(il);
    const bool tied = ggml_get_tensor(m->wctx_, "output.weight") == nullptr;
    const int G = (int)m->gpus_.size();                  // GPUs in use (0: CPU only)
    ggml_backend_buffer_type_t gpu_buft = G ? ggml_backend_get_default_buffer_type(m->main_) : nullptr;
    ggml_backend_buffer_type_t cpu_buft = cpu_buffer_type();
    std::vector<ggml_backend_buffer_type_t> bufts(G);
    std::vector<size_t> budgets(G, 0);                     // VRAM each GPU can give to weights and the KV cache
    {
        std::vector<size_t> forced;                        // tests: TIERLM_VRAM_BYTES=n[,n...] per GPU
        if (const char* e = std::getenv("TIERLM_VRAM_BYTES")) {
            std::string v = e;
            for (size_t q = 0; q <= v.size(); ) {
                size_t comma = v.find(',', q);
                if (comma == std::string::npos) comma = v.size();
                forced.push_back((size_t)std::strtoull(v.substr(q, comma - q).c_str(), nullptr, 10));
                q = comma + 1;
            }
        }
        for (int d = 0; d < G; ++d) {
            bufts[d] = ggml_backend_get_default_buffer_type(m->gpus_[d]);
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(m->gpus_[d]), &free, &total);
            size_t avail = free;
            if (opt.vram_budget_mb > 0) avail = std::min(avail, (size_t)opt.vram_budget_mb << 20);
            const size_t reserve = (size_t)768 << 20;      // compute scratch memory and driver overhead
            budgets[d] = avail > reserve ? avail - reserve : 0;
            if (!forced.empty()) budgets[d] = forced[std::min((size_t)d, forced.size() - 1)];
        }
    }
    size_t budget = G ? budgets[0] : 0;
    struct Group { int prio, layer; std::vector<ggml_tensor*> t; size_t bytes = 0; };
    std::vector<Group> groups;
    // "auto": mixture-of-experts models keep attention (and DeltaNet) on the GPU with the experts in RAM, since
    // the CPU then only reads the few experts each token picks; dense models move whole layers.
    const bool by_layer = opt.split_mode == "auto" ? !hp.is_moe() : opt.split_mode != "experts";
    m->place_.mode = by_layer ? "layers" : "experts";
    auto find_group = [&](int prio, int layer) -> Group& {
        for (auto& gr : groups) if (gr.prio == prio && gr.layer == layer) return gr;
        groups.push_back({prio, layer, {}, 0});
        return groups.back();
    };
    for (ggml_tensor* t = ggml_get_first_tensor(m->wctx_); t; t = ggml_get_next_tensor(m->wctx_, t)) {
        const std::string name = ggml_get_name(t);
        int prio = 1, layer = -1;
        bool own = false;                                                // a group of its own
        if (name == "token_embd.weight") { prio = tied ? 1 : 9; own = true; }   // only looked up by row: fine in RAM
        else if (name == "output.weight") { prio = 1; layer = hp.n_layer; own = true; }
        else if (name.compare(0, 4, "blk.") == 0) {
            layer = std::atoi(name.c_str() + 4);
            if (layer >= nl_all) continue;                               // draft (MTP) blocks: only when speculating
            const std::string rest = name.substr(name.find('.', 4) + 1);
            if (by_layer) prio = 2;                                      // the whole layer moves together
            else if (rest.find("_exps") != std::string::npos) prio = 4;
            else if (rest.compare(0, 9, "ffn_gate.") == 0 || rest.compare(0, 7, "ffn_up.") == 0 || rest.compare(0, 9, "ffn_down.") == 0) prio = 3;
            else own = true;
        } else { prio = 0; own = true; }
        Group& gr = own ? (groups.push_back({prio, layer, {}, 0}), groups.back()) : find_group(prio, layer);
        gr.t.push_back(t);
        if (gpu_buft) gr.bytes += GGML_PAD(ggml_backend_buft_get_alloc_size(gpu_buft, t), ggml_backend_buft_get_alignment(gpu_buft));
    }
    // Layer mode fills VRAM from the last layer backwards, so the hand-off to the GPU happens once, and the
    // output layer (also in VRAM) follows straight on.
    std::stable_sort(groups.begin(), groups.end(), [&](const Group& a, const Group& b) {
        if (a.prio != b.prio) return a.prio < b.prio;
        return a.prio == 2 ? a.layer > b.layer : a.layer < b.layer;
    });
    std::vector<ggml_tensor*> on_gpu, on_cpu;
    std::vector<bool> kv_gpu(nl_all, false);
    std::vector<std::vector<ggml_tensor*>> on_dev(G);       // per GPU
    std::vector<int> ldev(nl_all, -1);                      // per layer: the GPU holding its KV cache, or -1
    if (G <= 1) {
        bool kv_decided = false, layers_full = false;
        for (const Group& gr : groups) {
            if (!by_layer && gr.prio >= 3 && !kv_decided) {                  // the KV cache goes before the FFNs
                const bool fit = gpu_buft && kv_need <= budget;
                if (fit) budget -= kv_need;
                std::fill(kv_gpu.begin(), kv_gpu.end(), fit);
                kv_decided = true;
            }
            size_t need = gr.bytes + (by_layer && gr.prio == 2 && gr.layer >= 0 && gr.layer < nl_all ? layer_mem(gr.layer) : 0);
            bool gpu = gpu_buft && gr.prio < 9 && need <= budget && !(gr.prio == 2 && layers_full);
            if (gr.prio == 2 && !gpu) layers_full = true;                    // keep the VRAM layers contiguous
            if (gpu) budget -= need;
            if (gpu && by_layer && gr.prio == 2 && gr.layer >= 0 && gr.layer < nl_all) kv_gpu[gr.layer] = true;
            for (ggml_tensor* t : gr.t) (gpu ? on_gpu : on_cpu).push_back(t);
        }
        if (!by_layer && !kv_decided) {
            const bool fit = gpu_buft && kv_need <= budget;
            std::fill(kv_gpu.begin(), kv_gpu.end(), fit);
        }
        if (G == 1) { on_dev[0] = on_gpu; for (int i = 0; i < nl_all; ++i) ldev[i] = kv_gpu[i] ? 0 : -1; }
    } else if (by_layer) {
        // Several GPUs, whole layers: from the last layer backwards, fill the first GPU, then the next, then RAM.
        // The output layer and the small global tensors go on the first GPU, next to the last layers.
        int cur = 0;
        for (const Group& gr : groups) {
            const bool layer_group = gr.prio == 2 && gr.layer >= 0 && gr.layer < nl_all;
            const size_t need = gr.bytes + (layer_group ? layer_mem(gr.layer) : 0);
            int d = -1;
            if (gr.prio == 2) { while (cur < G && need > budgets[cur]) ++cur; if (cur < G) d = cur; }   // contiguous
            else if (gr.prio < 9) for (int k = 0; k < G && d < 0; ++k) if (need <= budgets[k]) d = k;
            if (d >= 0) { budgets[d] -= need; for (ggml_tensor* t : gr.t) on_dev[d].push_back(t); if (layer_group) ldev[gr.layer] = d; }
            else for (ggml_tensor* t : gr.t) on_cpu.push_back(t);
        }
    } else {
        // Several GPUs, experts in RAM: each layer's attention, router and KV cache go on a GPU (from the last
        // layer backwards, filling one GPU, then the next), then the experts: on their layer's GPU if it has room,
        // else on another GPU, else in RAM.
        std::vector<size_t> core(nl_all, 0);
        for (const Group& gr : groups) if (gr.prio < 3 && gr.layer >= 0 && gr.layer < nl_all) core[gr.layer] += gr.bytes;
        int cur = 0;
        for (int il = nl_all - 1; il >= 0; --il) {
            const size_t need = core[il] + layer_mem(il);
            while (cur < G && need > budgets[cur]) ++cur;
            if (cur == G) break;                                     // the remaining layers stay in RAM
            ldev[il] = cur;
            budgets[cur] -= need;
        }
        for (const Group& gr : groups) {
            int d = -1;
            if (gr.prio >= 9) d = -1;
            else if (gr.prio < 3 && gr.layer >= 0 && gr.layer < nl_all) d = ldev[gr.layer];   // already counted
            else {
                const int pref = gr.layer >= 0 && gr.layer < nl_all ? ldev[gr.layer] : 0;
                if (pref >= 0 && gr.bytes <= budgets[pref]) d = pref;
                for (int k = 0; k < G && d < 0; ++k) if (gr.bytes <= budgets[k]) d = k;
                if (d >= 0) budgets[d] -= gr.bytes;
            }
            for (ggml_tensor* t : gr.t) (d >= 0 ? on_dev[d] : on_cpu).push_back(t);
        }
    }
    if (G > 1) for (int i = 0; i < nl_all; ++i) kv_gpu[i] = ldev[i] >= 0;
    for (bool k : kv_gpu) m->place_.kv_gpu_layers += k;
    m->place_.kv_on_gpu = gpu_buft && m->place_.kv_gpu_layers == nl_all;
    m->layer_kv_gpu_ = kv_gpu;
    m->layer_dev_ = ldev;
    auto alloc = [&](ggml_backend_buffer_type_t buft, std::vector<ggml_tensor*>& ts, uint64_t& bytes) {
        if (ts.empty()) return;
        ggml_backend_buffer* b = ggml_backend_buft_alloc_buffer_n(buft, ts.data(), (int)ts.size());
        if (!b) throw std::runtime_error(std::string("not enough memory for the weights in ") + ggml_backend_buft_name(buft));
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bytes += ggml_backend_buffer_get_size(b);
        m->wbufs_.push_back(b);
    };
    // Weights the CPU computes with are stored in ggml's repacked layout where it has one (Q4_0, Q4_K, IQ4_NL, and
    // Q2_K with AVX-512, on x86): blocks of several rows interleaved, which its matrix kernels read faster. Only for
    // weights used in matrix multiplies (not the embedding table, which is looked up by row). TIERLM_NO_REPACK=1 or
    // LoadOptions::repack = false keeps the plain layout.
    std::vector<ggml_tensor*> on_repack;
    ggml_backend_buffer_type_t repack_buft = nullptr;
    if (opt.repack && !std::getenv("TIERLM_NO_REPACK")) {
        ggml_backend_dev_t cdev = ggml_backend_get_device(m->cpu_);
        auto get_extra = (ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(cdev), "ggml_backend_dev_get_extra_bufts");
        for (ggml_backend_buffer_type_t* p = get_extra ? get_extra(cdev) : nullptr; p && *p; ++p)
            if (std::string(ggml_backend_buft_name(*p)).find("REPACK") != std::string::npos) repack_buft = *p;
        ggml_backend_buffer_t probe = repack_buft ? ggml_backend_buft_alloc_buffer(repack_buft, 0) : nullptr;
        if (probe) {
            ggml_init_params tp{ 16 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
            ggml_context* tc = ggml_init(tp);
            std::vector<ggml_tensor*> keep;
            for (ggml_tensor* t : on_cpu) {
                const std::string n = ggml_get_name(t);
                const bool matmul = ggml_n_dims(t) >= 2 && t->ne[1] > 1 && n.find("embd") == std::string::npos &&
                                    n.find("embed") == std::string::npos && n.find("norm") == std::string::npos &&
                                    n.find("conv1d") == std::string::npos;
                bool ok = false;
                if (matmul) {                             // would the CPU engine run a multiply with it repacked?
                    ggml_reset(tc);
                    ggml_tensor* w = ggml_dup_tensor(tc, t);
                    w->buffer = probe;
                    ggml_tensor* op;
                    if (t->ne[2] > 1) {                   // experts
                        ggml_tensor* x = ggml_new_tensor_3d(tc, GGML_TYPE_F32, t->ne[0], 1, 8);
                        ggml_tensor* ids = ggml_new_tensor_2d(tc, GGML_TYPE_I32, 1, 8);
                        op = ggml_mul_mat_id(tc, w, x, ids);
                    } else {
                        op = ggml_mul_mat(tc, w, ggml_new_tensor_2d(tc, GGML_TYPE_F32, t->ne[0], 8));
                    }
                    ok = ggml_backend_dev_supports_op(cdev, op);
                }
                (ok ? on_repack : keep).push_back(t);
            }
            on_cpu.swap(keep);
            ggml_free(tc);
            ggml_backend_buffer_free(probe);
        }
    }
    try {
        m->place_.gpus.resize(G);
        for (int d = 0; d < G; ++d) {
            m->place_.gpus[d].name = ggml_backend_dev_name(ggml_backend_get_device(m->gpus_[d]));
            alloc(bufts[d], on_dev[d], m->place_.gpus[d].weights);
            m->place_.gpu_weights += m->place_.gpus[d].weights;
        }
        alloc(cpu_buft, on_cpu, m->place_.cpu_weights);
        if (!on_repack.empty()) alloc(repack_buft, on_repack, m->place_.cpu_weights);
    } catch (...) { gguf_free(g); throw; }
    m->place_.repacked = (int)on_repack.size();
    m->weights_bytes_ = m->place_.gpu_weights + m->place_.cpu_weights;
    m->place_.layers = hp.n_layer;
    if (G) {
        std::set<const ggml_tensor*> vram;
        for (const auto& v : on_dev) vram.insert(v.begin(), v.end());
        for (int i = 0; i < hp.n_layer; ++i) {
            const std::string b = "blk." + std::to_string(i) + ".";
            auto in_vram = [&](const char* n) {
                ggml_tensor* t = ggml_get_tensor(m->wctx_, (b + n).c_str());
                return t && vram.count(t) > 0;
            };
            m->place_.attn_gpu += in_vram("attn_norm.weight");
            const bool f = hp.n_expert > 0 ? in_vram("ffn_down_exps.weight") : in_vram("ffn_up.weight");
            m->place_.ffn_gpu += f;
            m->place_.ffn_vram.push_back(f);
        }
    }

    // ---- stream every tensor's data from its file into wherever the tier loader put it ----
    std::vector<std::ifstream> files;
    for (const auto& p : paths) {
        files.emplace_back(std::filesystem::u8path(p), std::ios::binary);   // UTF-8 path, also on Windows
        if (!files.back()) { gguf_free(g); throw std::runtime_error("cannot open " + p); }
    }
    std::vector<char> chunk;
    size_t total_bytes = 0, done_bytes = 0, next_report = 0;
    for (const Source& s : sources) if (s.t->buffer) total_bytes += ggml_nbytes(s.t);
    for (const Source& s : sources) {
        if (!s.t->buffer) continue;                                    // not loaded (draft blocks)
        const size_t n = ggml_nbytes(s.t);
        done_bytes += n;
        if (opt.progress && (done_bytes >= next_report || done_bytes == total_bytes)) {
            opt.progress(total_bytes ? (float)((double)done_bytes / total_bytes) : 1.0f);
            next_report = done_bytes + total_bytes / 200;
        }
        chunk.resize(n);
        auto& f = files[s.part];
        f.seekg((std::streamoff)s.offset);
        if (!f.read(chunk.data(), (std::streamsize)n)) { gguf_free(g); throw std::runtime_error("short read on " + paths[s.part]); }
        ggml_backend_tensor_set(s.t, chunk.data(), 0, n);
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

    auto bind_layer = [&](int i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        Layer L;
        L.attn_norm = T(b + "attn_norm.weight");
        L.recurrent = hp.is_recurrent(i);
        if (L.recurrent) {                                            // Gated DeltaNet layer
            if (hp.v_grouped) {                                       // Qwen3-Next
                L.wqkv = T(b + "attn_qkv.weight", false);
                if (L.wqkv) L.wgate = T(b + "attn_gate.weight");
                else L.ssm_in = T(b + "ssm_in.weight");               // older files: q, k, v, z interleaved per key head
                L.ssm_ba = T(b + "ssm_ba.weight");                    // beta and alpha, interleaved per key head
            } else {
                L.wqkv      = T(b + "attn_qkv.weight");
                L.wgate     = T(b + "attn_gate.weight");
                L.ssm_beta  = T(b + "ssm_beta.weight");
                L.ssm_alpha = T(b + "ssm_alpha.weight");
            }
            L.ssm_conv1d = T(b + "ssm_conv1d.weight");
            L.ssm_dt     = T(b + "ssm_dt.bias");
            L.ssm_a      = T(b + "ssm_a");
            L.ssm_norm   = T(b + "ssm_norm.weight");
            L.ssm_out    = T(b + "ssm_out.weight");
        } else {
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
        }
        L.ffn_norm  = T(b + (hp.hybrid ? "post_attention_norm.weight" : "ffn_norm.weight"));
        if (hp.is_moe()) {
            L.router = T(b + "ffn_gate_inp.weight");
            L.gate_up_x = T(b + "ffn_gate_up_exps.weight", false);      // gate and up stored together (newer files)
            if (!L.gate_up_x) {
                L.gate_x = T(b + "ffn_gate_exps.weight", false);
                if (!L.gate_x) throw std::runtime_error("this file stores experts as separate tensors (an old Mixtral GGUF); re-download a newer conversion");
                L.up_x   = T(b + "ffn_up_exps.weight");
            }
            L.down_x = T(b + "ffn_down_exps.weight");
            // Shared expert (Qwen3.5/3.6 MoE): runs on every token, scaled by its own sigmoid gate.
            L.up_s = T(b + "ffn_up_shexp.weight", false);
            if (L.up_s) {
                L.gate_s     = T(b + "ffn_gate_shexp.weight");
                L.down_s     = T(b + "ffn_down_shexp.weight");
                L.gate_inp_s = T(b + "ffn_gate_inp_shexp.weight");
            }
        } else {
            L.gate = T(b + "ffn_gate.weight");
            L.up   = T(b + "ffn_up.weight");
            L.down = T(b + "ffn_down.weight");
        }
        return L;
    };
    for (int i = 0; i < hp.n_layer; ++i) m->layers_.push_back(bind_layer(i));
    if (want_mtp) {
        // The MTP block: a full-attention layer like the model's own, plus its input projection: it combines the
        // next token's embedding with the model's last hidden state.
        const std::string b = "blk." + std::to_string(hp.n_layer) + ".";
        Layer L = bind_layer(hp.n_layer);
        L.eh_proj   = T(b + "nextn.eh_proj.weight");
        L.enorm     = T(b + "nextn.enorm.weight");
        L.hnorm     = T(b + "nextn.hnorm.weight");
        L.mtp_embd  = T(b + "nextn.embed_tokens.weight", false);     // usually shared with the model
        L.head_norm = T(b + "nextn.shared_head_norm.weight", false);
        L.head      = T(b + "nextn.shared_head_head.weight", false);
        m->mtp_ = std::make_unique<Layer>(L);
    }

    // ---- KV cache on the same device, in f16: half the size and memory traffic of f32 ----
    const int hd = hp.head_dim;
    const ggml_type kv_type = (ggml_type)m->kv_t_;
    ggml_init_params kp{ (size_t)2 * nl_all * ggml_tensor_overhead() + 1024, nullptr, /*no_alloc=*/true };
    m->kvctx_ = ggml_init(kp);
    std::vector<std::vector<ggml_tensor*>> kv_t(G + 1);              // per GPU, then RAM
    for (int i = 0; i < nl_all; ++i) {
        auto& L = i < hp.n_layer ? m->layers_[i] : *m->mtp_;
        auto& dst = kv_t[m->layer_dev_[i] >= 0 ? m->layer_dev_[i] : G];
        if (L.recurrent) {                                            // DeltaNet: fixed-size state, always f32
            L.conv_state  = ggml_new_tensor_2d(m->kvctx_, GGML_TYPE_F32, hp.ssm_conv - 1, hp.ssm_conv_dim());
            L.ssm_state_t = ggml_new_tensor_3d(m->kvctx_, GGML_TYPE_F32, hp.ssm_state, hp.ssm_state, hp.ssm_v_heads);
            dst.push_back(L.conv_state);
            dst.push_back(L.ssm_state_t);
            continue;
        }
        L.k_cache = ggml_new_tensor_3d(m->kvctx_, kv_type, hd, hp.n_head_kv, m->max_ctx_);
        L.v_cache = ggml_new_tensor_3d(m->kvctx_, kv_type, hd, hp.n_head_kv, m->max_ctx_);
        dst.push_back(L.k_cache);
        dst.push_back(L.v_cache);
    }
    for (int tier = 0; tier <= G; ++tier) {
        auto& ts = kv_t[tier];
        if (ts.empty()) continue;
        ggml_backend_buffer* b = ggml_backend_buft_alloc_buffer_n(tier < G ? bufts[tier] : cpu_buft, ts.data(), (int)ts.size());
        if (!b) throw std::runtime_error("not enough memory for the KV cache (try a smaller context)");
        ggml_backend_buffer_clear(b, 0);
        m->kv_bytes_ += ggml_backend_buffer_get_size(b);
        if (tier < G) m->place_.gpus[tier].kv += ggml_backend_buffer_get_size(b);
        m->kvbufs_.push_back(b);
    }
    for (int i = 0; i < hp.n_layer; ++i) if (m->layer_dev_[i] >= 0) m->place_.gpus[m->layer_dev_[i]].layers++;

    // Fused flash attention reads the cache once, with no copies. TIERLM_NO_FA=1 uses the plain version.
    m->flash_attn_ = opt.flash_attn >= 0 ? opt.flash_attn == 1 : std::getenv("TIERLM_NO_FA") == nullptr;
    if (ggml_is_quantized((ggml_type)m->kv_t_)) m->flash_attn_ = true;               // only flash attention reads a quantized cache

    // ---- scheduler: runs each op on the device, falling back to the CPU where needed ----
    std::vector<ggml_backend_t> backends(m->gpus_.begin(), m->gpus_.end());
    backends.push_back(m->cpu_);
    const int n_backends = (int)backends.size();
    // With weights split between VRAM and RAM, let the scheduler copy RAM weights to the GPU for big
    // batches (reading a long prompt), where that beats computing them on the CPU.
    const bool offload = m->main_ != m->cpu_ && (m->place_.cpu_weights > 0 || !m->place_.kv_on_gpu);
    // Layer split: pin each layer's ops to its device (RAM layers to the CPU; with several GPUs, each layer to its
    // GPU too, so ops without weights don't wander to another card).
    m->pin_layers_ = m->main_ != m->cpu_ && m->place_.mode == "layers" && (m->place_.kv_gpu_layers < hp.n_layer || G > 1);
    m->sched_ = ggml_backend_sched_new(backends.data(), nullptr, n_backends, kGraphSize, false, offload);
    if (!m->sched_) throw std::runtime_error("failed to create the scheduler");
    m->graph_mem_.resize(ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false));

    // Reserve compute memory once, for the largest graph we will ever build (a full batch at the end
    // of the context), so it is not re-planned on every token.
    const int n_res = std::min(m->batch_, m->max_ctx_);
    Graph worst = m->build(n_res, m->max_ctx_ - n_res, true);
    m->pin_cpu_layers(worst);
    bool ok = ggml_backend_sched_reserve(m->sched_, worst.gf);
    ggml_free(worst.ctx);
    if (!ok) throw std::runtime_error("failed to reserve compute memory");
    return m;
}

// ggml's scheduler gives any op without weights of its own (norms, RoPE, attention over the cache...) to the
// GPU if it can run it, even in the middle of a layer whose weights are in RAM. Each such op costs two
// GPU<->CPU hand-offs, which on Windows cost more than the GPU saves. So in a layer split, every op of a RAM
// layer is pinned to the CPU, and the work changes device once per token.
void Model::pin_cpu_layers(const Graph& g) {
    // Debugging: TIERLM_CPU_OPS=SSM_CONV,GATED_DELTA_NET,... runs those op types on the CPU, to find which
    // GPU kernel disagrees with the reference.
    if (const char* ops = std::getenv("TIERLM_CPU_OPS"); ops && main_ != cpu_) {
        const std::string list = std::string(",") + ops + ",";
        for (int i = 0; i < ggml_graph_n_nodes(g.gf); ++i) {
            ggml_tensor* t = ggml_graph_node(g.gf, i);
            if (list.find(std::string(",") + ggml_op_name(t->op) + ",") != std::string::npos)
                ggml_backend_sched_set_tensor_backend(sched_, t, cpu_);
        }
    }
    if (!pin_layers_) return;
    for (const auto& [t, layer] : g.owner) {
        if (layer < 0 || layer >= (int)layer_dev_.size()) continue;
        const int d = layer_dev_[layer];
        if (d < 0) ggml_backend_sched_set_tensor_backend(sched_, t, cpu_);
        else if (gpus_.size() > 1) ggml_backend_sched_set_tensor_backend(sched_, t, gpus_[d]);
    }
}

void Model::reset() {
    release_pending();
    n_past_ = 0;
    if (hp_.hybrid) for (auto* b : kvbufs_) ggml_backend_buffer_clear(b, 0);   // clear the running memory
}

void Model::truncate(int n) {
    release_pending();
    if (n >= n_past_) return;
    if (hp_.hybrid) { reset(); return; }                        // the running memory can't be rewound
    n_past_ = n < 0 ? 0 : n;
}

const char* Model::kv_type() const { return ggml_type_name((ggml_type)kv_t_); }

bool Model::integrated_gpu() const {
    return dev_ && ggml_backend_dev_type((ggml_backend_dev_t)dev_) == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

void Model::device_memory(size_t& free, size_t& total) const {
    free = total = 0;
    if (dev_) ggml_backend_dev_memory((ggml_backend_dev_t)dev_, &free, &total);
}

// Full attention for one layer: projections, QK-norm, RoPE, writing this batch's keys and values into the layer's
// cache at positions [n_past, n_past + n_tok), then attention over everything cached so far. Returns the output
// projection (to be added to the residual stream). Shared by the model's layers and the MTP draft head.
ggml_tensor* Model::attn_block(ggml_context* c, Graph& g, const Layer& L, ggml_tensor* h, int n_tok, int n_past) {
    const Hparams& hp = hp_;
    const int hd = hp.head_dim, n_kv = n_past + n_tok;
    const float scale = 1.0f / std::sqrt((float)hd);
        // Qwen3.5's attention layers project each head's query together with an output gate: [q | gate]
        ggml_tensor* qg = hp.hybrid ? ggml_mul_mat(c, L.wq, h) : nullptr;    // [2*hd*n_head, n_tok]
        ggml_tensor* q = qg ? ggml_view_3d(c, qg, hd, hp.n_head, n_tok, 2 * hd * ggml_element_size(qg),
                                           2 * hd * hp.n_head * ggml_element_size(qg), 0)
                            : ggml_mul_mat(c, L.wq, h);                       // [n_head*hd, n_tok]
        ggml_tensor* k = ggml_mul_mat(c, L.wk, h);
        ggml_tensor* v = ggml_mul_mat(c, L.wv, h);
        if (L.bq) { q = ggml_add(c, q, L.bq); k = ggml_add(c, k, L.bk); v = ggml_add(c, v, L.bv); }
        if (hp.qk_norm == Hparams::kFull) {                                  // OLMoE: over the full width
            q = ggml_mul(c, ggml_rms_norm(c, q, hp.rms_eps), L.q_norm);
            k = ggml_mul(c, ggml_rms_norm(c, k, hp.rms_eps), L.k_norm);
        }
        if (!qg) q = ggml_reshape_3d(c, q, hd, hp.n_head, n_tok);
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
        if (qg) {                                                            // output gate (Qwen3.5)
            ggml_tensor* gate = ggml_cont_2d(c, ggml_view_3d(c, qg, hd, hp.n_head, n_tok, 2 * hd * ggml_element_size(qg),
                                                             2 * hd * hp.n_head * ggml_element_size(qg), hd * ggml_element_size(qg)),
                                             hd * hp.n_head, n_tok);
            o = ggml_mul(c, o, ggml_sigmoid(c, gate));
        }
        return ggml_mul_mat(c, L.wo, o);
}

// Feed-forward for one layer: dense SwiGLU, or mixture of experts (with an optional shared expert).
// Returns the output to be added to the residual stream.
ggml_tensor* Model::ffn_block(ggml_context* c, Graph& g, const Layer& L, ggml_tensor* h, int n_tok, bool want_routing) {
    const Hparams& hp = hp_;
        if (!hp.is_moe()) {
            // ---- dense SwiGLU feed-forward ----
            ggml_tensor* act = ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, L.gate, h)), ggml_mul_mat(c, L.up, h));
            return ggml_mul_mat(c, L.down, act);
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
        if (hp.expert_scale != 0.0f && hp.expert_scale != 1.0f) w = ggml_scale(c, w, hp.expert_scale);
        ggml_tensor* h3    = ggml_reshape_3d(c, h, hp.n_embd, 1, n_tok);
        ggml_tensor *up, *gate;
        if (L.gate_up_x) {                                                   // [2*n_ff_exp, n_used, n_tok]: gate, then up
            ggml_tensor* gu = ggml_mul_mat_id(c, L.gate_up_x, h3, sel);
            const int64_t f = gu->ne[0] / 2;
            gate = ggml_view_3d(c, gu, f, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], 0);
            up   = ggml_view_3d(c, gu, f, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], f * gu->nb[0]);
        } else {
            up   = ggml_mul_mat_id(c, L.up_x, h3, sel);                      // [n_ff_exp, n_used, n_tok]
            gate = ggml_mul_mat_id(c, L.gate_x, h3, sel);
        }
        ggml_tensor* act   = L.gate_up_x ? ggml_swiglu_split(c, gate, up)   // works on the two halves in place
                                     : ggml_mul(c, ggml_silu(c, gate), up);
        ggml_tensor* down  = ggml_mul_mat_id(c, L.down_x, act, sel);         // [n_embd, n_used, n_tok]
        down = ggml_mul(c, down, w);                                         // weight each expert's output
        ggml_tensor* moe = nullptr;
        for (int e = 0; e < hp.n_expert_used; ++e) {
            ggml_tensor* s = ggml_view_2d(c, down, hp.n_embd, n_tok, down->nb[2], e * down->nb[1]);
            moe = moe ? ggml_add(c, moe, s) : s;
        }
        if (L.up_s) {                                                        // shared expert
            ggml_tensor* sh = ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, L.gate_s, h)), ggml_mul_mat(c, L.up_s, h));
            sh = ggml_mul_mat(c, L.down_s, sh);                              // [n_embd, n_tok]
            ggml_tensor* sg = ggml_sigmoid(c, ggml_mul_mat(c, L.gate_inp_s, h));   // [1, n_tok]
            moe = ggml_add(c, moe, ggml_mul(c, sh, sg));
        }
        return moe;
}

// Builds the compute graph for n_tok new tokens placed after n_past cached ones.
// Only a description: the scheduler allocates its memory on the right device later.
Model::Graph Model::build(int n_tok, int n_past, bool want_routing, int n_logits, bool want_hidden, int snapshots) {
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
    // Remember which layer created each tensor, so a layer split can keep RAM layers wholly on the CPU.
    ggml_tensor* mark = nullptr;
    auto tag = [&](int layer) {
        for (ggml_tensor* t = mark ? ggml_get_next_tensor(c, mark) : ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) {
            if (layer >= 0 && pin_layers_) g.owner.push_back({t, layer});
            mark = t;
        }
    };
    for (int li = 0; li < (int)layers_.size(); ++li) {
        tag(li - 1);
        const Layer& L = layers_[li];
        // ---- attention ----
        ggml_tensor* h = ggml_mul(c, ggml_rms_norm(c, x, hp.rms_eps), L.attn_norm);
        if (L.recurrent) {
            // ---- Gated DeltaNet (linear attention with a running memory) ----
            const int S = hp.ssm_state, Hk = hp.ssm_k_heads, Hv = hp.ssm_v_heads, CD = hp.ssm_conv_dim(), KC = hp.ssm_conv;
            const int R = Hv / Hk;                                               // value heads per key head
            ggml_tensor *qkv, *z;
            if (L.ssm_in) {
                // Older Qwen3-Next files: per key head [q | k | R values | R gates]. Gather into [all q | all k | all v].
                ggml_tensor* m  = ggml_reshape_3d(c, ggml_mul_mat(c, L.ssm_in, h), 2 * S + 2 * R * S, Hk, n_tok);
                const size_t e = ggml_element_size(m);
                auto part = [&](int64_t w, int64_t off) {
                    return ggml_cont_2d(c, ggml_view_3d(c, m, w, Hk, n_tok, m->nb[1], m->nb[2], (size_t)off * e), w * Hk, n_tok);
                };
                qkv = ggml_concat(c, ggml_concat(c, part(S, 0), part(S, S), 0), part((int64_t)R * S, 2 * S), 0);   // [CD, n_tok]
                z   = part((int64_t)R * S, 2 * S + (int64_t)R * S);                                                 // [S*Hv, n_tok]
            } else {
                qkv = ggml_mul_mat(c, L.wqkv, h);                                // [CD, n_tok]
                z   = ggml_mul_mat(c, L.wgate, h);                               // [S*Hv, n_tok]
            }
            ggml_tensor *b_in, *a_in;                                            // [Hv, n_tok] each
            if (L.ssm_ba) {                                                      // Qwen3-Next: per key head [R betas | R alphas]
                ggml_tensor* ba = ggml_reshape_3d(c, ggml_mul_mat(c, L.ssm_ba, h), 2 * R, Hk, n_tok);
                const size_t e = ggml_element_size(ba);
                b_in = ggml_cont_2d(c, ggml_view_3d(c, ba, R, Hk, n_tok, ba->nb[1], ba->nb[2], 0), Hv, n_tok);
                a_in = ggml_cont_2d(c, ggml_view_3d(c, ba, R, Hk, n_tok, ba->nb[1], ba->nb[2], (size_t)R * e), Hv, n_tok);
            } else {
                b_in = ggml_mul_mat(c, L.ssm_beta, h);
                a_in = ggml_mul_mat(c, L.ssm_alpha, h);
            }
            ggml_tensor* beta = ggml_sigmoid(c, b_in);                           // how strongly to write, per head
            beta = ggml_reshape_4d(c, beta, 1, Hv, n_tok, 1);
            // log of the memory's decay: -exp(A_log) * softplus(alpha + dt_bias)  (ssm_a holds -exp(A_log))
            ggml_tensor* dec = ggml_mul(c, ggml_softplus(c, ggml_add(c, a_in, L.ssm_dt)), L.ssm_a);
            dec = ggml_reshape_4d(c, dec, 1, Hv, n_tok, 1);
            // Causal convolution over the last KC inputs: [the KC-1 saved from before | this batch]
            ggml_tensor* cs  = ggml_reshape_3d(c, L.conv_state, KC - 1, CD, 1);
            ggml_tensor* cin = ggml_concat(c, cs, ggml_transpose(c, ggml_reshape_3d(c, qkv, CD, n_tok, 1)), 0);
            ggml_build_forward_expand(g.gf, ggml_cpy(c, ggml_view_3d(c, cin, KC - 1, CD, 1, cin->nb[1], cin->nb[2],
                                                                     (size_t)n_tok * ggml_element_size(cin)), cs));
            ggml_tensor* co = ggml_silu(c, ggml_ssm_conv(c, cin, L.ssm_conv1d)); // [CD, n_tok, 1]
            const size_t es = ggml_element_size(co), rs = co->nb[1];
            ggml_tensor* qh = ggml_view_4d(c, co, S, Hk, n_tok, 1, S * es, rs, rs * n_tok, 0);
            ggml_tensor* kh = ggml_view_4d(c, co, S, Hk, n_tok, 1, S * es, rs, rs * n_tok, (size_t)S * Hk * es);
            ggml_tensor* vh = ggml_view_4d(c, co, S, Hv, n_tok, 1, S * es, rs, rs * n_tok, (size_t)2 * S * Hk * es);
            // l2-normalise q and k: x / sqrt(sum(x^2) + eps), written with rms_norm so every backend has it
            auto l2 = [&](ggml_tensor* t) { return ggml_scale(c, ggml_rms_norm(c, t, hp.rms_eps / S), 1.0f / std::sqrt((float)S)); };
            qh = l2(qh);
            kh = l2(kh);
            if (hp.v_grouped && R > 1) {
                // Value heads grouped by key head (head j uses key head j / R): repeat each key head R times in place.
                auto rep_heads = [&](ggml_tensor* t) {
                    t = ggml_repeat_4d(c, ggml_reshape_4d(c, t, S, 1, Hk, n_tok), S, R, Hk, n_tok);
                    return ggml_reshape_4d(c, t, S, Hv, n_tok, 1);
                };
                qh = rep_heads(qh);
                kh = rep_heads(kh);
            }
            ggml_tensor* st  = ggml_reshape_4d(c, L.ssm_state_t, S, S, Hv, 1);
            // Outputs, then the new memory. With snapshots > 1 the op also keeps the memory as it was 1, 2, ...
            // tokens back, so a speculative check can be undone for the tokens that weren't accepted.
            ggml_tensor* res = ggml_gated_delta_net(c, qh, kh, vh, dec, beta, st, snapshots);
            ggml_tensor* out = ggml_view_4d(c, res, S, Hv, n_tok, 1, (size_t)S * 4, (size_t)S * Hv * 4, (size_t)S * Hv * n_tok * 4, 0);
            ggml_tensor* ns  = ggml_view_1d(c, res, (int64_t)S * S * Hv, (size_t)S * Hv * n_tok * 4);
            ggml_build_forward_expand(g.gf, ggml_cpy(c, ns, ggml_view_1d(c, L.ssm_state_t, (int64_t)S * S * Hv, 0)));
            if (snapshots > 1) {
                // For dropping the last d tokens: the memory d tokens back, and the convolution's last inputs before
                // them. Kept as outputs until Model::accept() has used them.
                ggml_set_output(res);
                auto& rb = g.rollback[li];
                rb.resize(std::min(snapshots, n_tok));
                for (int d = 1; d < (int)rb.size(); ++d) {
                    rb[d].first = ggml_view_3d(c, res, S, S, Hv, (size_t)S * 4, (size_t)S * S * 4,
                                               ((size_t)S * Hv * n_tok + (size_t)d * S * S * Hv) * 4);
                    rb[d].second = ggml_cont(c, ggml_view_3d(c, cin, KC - 1, CD, 1, cin->nb[1], cin->nb[2],
                                                             (size_t)(n_tok - d) * ggml_element_size(cin)));
                    ggml_set_output(rb[d].second);
                    ggml_build_forward_expand(g.gf, rb[d].first);      // views get their memory only as graph nodes
                    ggml_build_forward_expand(g.gf, rb[d].second);
                }
            }
            // Gated norm: rms_norm(out) * w * silu(z), then back to the model width
            ggml_tensor* on = ggml_mul(c, ggml_mul(c, ggml_rms_norm(c, out, hp.rms_eps), L.ssm_norm),
                                       ggml_silu(c, ggml_reshape_4d(c, z, S, Hv, n_tok, 1)));
            on = ggml_reshape_2d(c, on, (int64_t)S * Hv, n_tok);
            x = ggml_add(c, x, ggml_mul_mat(c, L.ssm_out, on));
        } else {
            x = ggml_add(c, x, attn_block(c, g, L, h, n_tok, n_past));
        }

        h = ggml_mul(c, ggml_rms_norm(c, x, hp.rms_eps), L.ffn_norm);
        x = ggml_add(c, x, ffn_block(c, g, L, h, n_tok, want_routing));
    }
    tag((int)layers_.size() - 1);

    x = ggml_mul(c, ggml_rms_norm(c, x, hp.rms_eps), out_norm_);
    if (want_hidden) { g.hidden = x; ggml_set_output(x); }                  // [n_embd, n_tok]: what the MTP head reads
    n_logits = std::max(1, std::min(n_logits, n_tok));                     // predictions for the last n_logits positions
    ggml_tensor* last = ggml_view_2d(c, x, hp.n_embd, n_logits, x->nb[1], (size_t)(n_tok - n_logits) * x->nb[1]);
    g.logits = ggml_mul_mat(c, output_, last);                             // [n_vocab, n_logits]
    ggml_set_output(g.logits);
    ggml_build_forward_expand(g.gf, g.logits);
    return g;
}

// The MTP draft block (multi-token prediction, as trained with Qwen3.5 / 3.6 / 3.8 and Qwen3-Next). At position p
// it reads the token at p and the model's final hidden state at p - 1, and predicts the token at p + 1:
//   x = eh_proj([rms(embed(token)) * enorm ; rms(h) * hnorm]), then one full-attention layer with its own KV
//   cache, then the shared final norm and output head.
// Its own final hidden state can stand in for the model's, to draft several tokens ahead.
Model::Graph Model::build_mtp(int n_tok, int n_past) {
    const Hparams& hp = hp_;
    const Layer& L = *mtp_;
    const int n_kv = n_past + n_tok;
    Graph g;
    ggml_init_params ip{ graph_mem_.size(), graph_mem_.data(), /*no_alloc=*/true };
    ggml_context* c = g.ctx = ggml_init(ip);
    g.gf = ggml_new_graph_custom(c, kGraphSize, false);
    g.n_tok = n_tok;
    g.inp = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tok);  ggml_set_input(g.inp);
    g.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tok);  ggml_set_input(g.pos);
    g.h_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, hp.n_embd, n_tok);  ggml_set_input(g.h_in);
    if (flash_attn_) { g.mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv, n_tok); ggml_set_input(g.mask); }

    ggml_tensor* e = ggml_get_rows(c, L.mtp_embd ? L.mtp_embd : tok_embd_, g.inp);
    e = ggml_mul(c, ggml_rms_norm(c, e, hp.rms_eps), L.enorm);
    ggml_tensor* hn = ggml_mul(c, ggml_rms_norm(c, g.h_in, hp.rms_eps), L.hnorm);
    ggml_tensor* x = ggml_mul_mat(c, L.eh_proj, ggml_concat(c, e, hn, 0));  // [n_embd, n_tok]
    ggml_tensor* h = ggml_mul(c, ggml_rms_norm(c, x, hp.rms_eps), L.attn_norm);
    x = ggml_add(c, x, attn_block(c, g, L, h, n_tok, n_past));
    h = ggml_mul(c, ggml_rms_norm(c, x, hp.rms_eps), L.ffn_norm);
    x = ggml_add(c, x, ffn_block(c, g, L, h, n_tok, false));
    x = ggml_mul(c, ggml_rms_norm(c, x, hp.rms_eps), L.head_norm ? L.head_norm : out_norm_);
    g.hidden = x; ggml_set_output(x);
    ggml_tensor* last = ggml_view_2d(c, x, hp.n_embd, 1, x->nb[1], (size_t)(n_tok - 1) * x->nb[1]);
    g.logits = ggml_mul_mat(c, L.head ? L.head : output_, last);
    ggml_set_output(g.logits);
    ggml_build_forward_expand(g.gf, g.logits);
    return g;
}

std::vector<float> Model::mtp_eval(const int32_t* ids, const float* h, int n_tok, int pos0, int n_threads, std::vector<float>* h_out) {
    if (!mtp_) throw std::runtime_error("this model has no MTP draft block loaded");
    if (pos0 + n_tok > max_ctx_) throw std::runtime_error("MTP: past the end of the context");
    release_pending();
    const Hparams& hp = hp_;
    Graph g = build_mtp(n_tok, pos0);
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, g.gf)) { ggml_free(g.ctx); throw std::runtime_error("failed to allocate compute memory"); }
    std::vector<int32_t> positions(n_tok);
    for (int i = 0; i < n_tok; ++i) positions[i] = pos0 + i;
    ggml_backend_tensor_set(g.inp, ids, 0, sizeof(int32_t) * n_tok);
    ggml_backend_tensor_set(g.pos, positions.data(), 0, sizeof(int32_t) * n_tok);
    ggml_backend_tensor_set(g.h_in, h, 0, sizeof(float) * (size_t)hp.n_embd * n_tok);
    if (g.mask) {
        const int n_kv = pos0 + n_tok;
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask((size_t)n_kv * n_tok);
        for (int t = 0; t < n_tok; ++t)
            for (int j = 0; j < n_kv; ++j) mask[(size_t)t * n_kv + j] = j <= pos0 + t ? zero : ninf;
        ggml_backend_tensor_set(g.mask, mask.data(), 0, sizeof(ggml_fp16_t) * mask.size());
    }
    set_n_threads(cpu_, n_threads);
    if (ggml_backend_sched_graph_compute(sched_, g.gf) != GGML_STATUS_SUCCESS) { ggml_free(g.ctx); throw std::runtime_error("graph compute failed"); }
    std::vector<float> out(hp.n_vocab);
    ggml_backend_tensor_get(g.logits, out.data(), 0, sizeof(float) * hp.n_vocab);
    if (h_out) {
        h_out->resize((size_t)hp.n_embd * n_tok);
        ggml_backend_tensor_get(g.hidden, h_out->data(), 0, sizeof(float) * h_out->size());
    }
    ggml_free(g.ctx);
    return out;
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
    return eval_ex(ids, n_tok, n_threads, 1, false, 1, routing).logits;
}

// The graph of a speculative check, kept until accept() knows how many of its tokens to keep.
void Model::release_pending() {
    if (pending_) { ggml_free(((Graph*)pending_)->ctx); delete (Graph*)pending_; pending_ = nullptr; }
}

void Model::accept(int n_keep) {
    if (!pending_) return;
    Graph& g = *(Graph*)pending_;
    const int d = g.n_tok - n_keep;                               // tokens to drop
    if (d > 0) {
        n_past_ -= d;                                             // attention layers: just forget their keys/values
        for (auto& [li, rb] : g.rollback) {                       // DeltaNet layers: restore the memory d tokens back
            if (d >= (int)rb.size()) throw std::runtime_error("speculative rollback deeper than its snapshots");
            Layer& L = layers_[li];
            ggml_backend_tensor_copy(rb[d].first, L.ssm_state_t);
            ggml_backend_tensor_copy(rb[d].second, L.conv_state);
        }
    }
    release_pending();
}

Model::EvalOut Model::eval_ex(const int32_t* ids, int n_tok, int n_threads, int n_logits, bool want_hidden, int snapshots,
                              std::vector<int32_t>* routing) {
    release_pending();                                            // an unanswered check counts as fully accepted
    const Hparams& hp = hp_;
    const int n_past = n_past_, n_kv = n_past + n_tok;
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    auto t0 = clk::now();
    Graph g = build(n_tok, n_past, routing != nullptr, n_logits, want_hidden, snapshots);
    g.n_tok = n_tok;
    auto t1 = clk::now();

    ggml_backend_sched_reset(sched_);
    pin_cpu_layers(g);
    if (!ggml_backend_sched_alloc_graph(sched_, g.gf)) { ggml_free(g.ctx); throw std::runtime_error("failed to allocate compute memory"); }
    auto t2 = clk::now();
    if (std::getenv("TIERLM_DUMP_SCHED") && n_tok == 1) {           // debugging: where the work switches device
        ggml_backend_t prev = nullptr;
        int layer = -1, shown = 0;
        for (int i = 0; i < ggml_graph_n_nodes(g.gf) && shown < 60; ++i) {
            ggml_tensor* t = ggml_graph_node(g.gf, i);
            if (t->op == GGML_OP_FLASH_ATTN_EXT || (t->op == GGML_OP_SOFT_MAX && !flash_attn_)) ++layer;
            ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched_, t);
            if (b == prev) continue;
            prev = b;
            ++shown;
            std::string srcs;
            for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
                ggml_tensor* x = t->src[s];
                ggml_backend_buffer* buf = x->view_src ? x->view_src->buffer : x->buffer;
                srcs += std::string(" [") + ggml_op_desc(x) + (buf ? std::string(" in ") + ggml_backend_buffer_name(buf) : "") + "]";
            }
            std::fprintf(stderr, "switch at node %4d (layer ~%d): %-14s -> %-8s inputs:%s\n", i, layer, ggml_op_desc(t),
                         b ? ggml_backend_name(b) : "?", srcs.c_str());
        }
        std::fprintf(stderr, "splits %d\n", ggml_backend_sched_get_n_splits(sched_));
    }

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
    set_n_threads(cpu_, n_threads);
    auto t3 = clk::now();

    if (ggml_backend_sched_graph_compute(sched_, g.gf) != GGML_STATUS_SUCCESS) {
        ggml_free(g.ctx);
        throw std::runtime_error("graph compute failed");
    }
    auto t4 = clk::now();
    EvalOut res;
    res.logits.resize((size_t)hp.n_vocab * g.logits->ne[1]);
    ggml_backend_tensor_get(g.logits, res.logits.data(), 0, sizeof(float) * res.logits.size());
    if (g.hidden) {
        res.hidden.resize((size_t)hp.n_embd * n_tok);
        ggml_backend_tensor_get(g.hidden, res.hidden.data(), 0, sizeof(float) * res.hidden.size());
    }
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
    if (snapshots > 1) { pending_ = new Graph(g); }              // kept for accept()
    else ggml_free(g.ctx);
    auto t5 = clk::now();
    stats_.calls++; stats_.tokens += n_tok;
    stats_.build_ms += ms(t0, t1); stats_.alloc_ms += ms(t1, t2); stats_.input_ms += ms(t2, t3);
    stats_.compute_ms += ms(t3, t4); stats_.output_ms += ms(t4, t5);
    stats_.splits = ggml_backend_sched_get_n_splits(sched_); stats_.copies = ggml_backend_sched_get_n_copies(sched_);
    n_past_ = n_kv;
    return res;
}

}
