#include "model.h"
#include "gguf_split.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "backends.h"
#include "gguf.h"
#include <algorithm>
#include <atomic>
#include <set>
#include <functional>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace tierlm {

static const char* kArchs[] = { "llama", "qwen2", "qwen3", "qwen3moe", "olmoe", "qwen35", "qwen35moe", "qwen3next", "gpt-oss", "flint", "deepseek2", "qwen4exp", "nemotron_h_moe" };
bool arch_supported(const std::string& arch) {
    for (const char* a : kArchs) if (arch == a) return true;
    return false;
}
std::vector<std::string> supported_archs() { return std::vector<std::string>(std::begin(kArchs), std::end(kArchs)); }

static constexpr int kGraphSize = 16384;

Model::~Model() {
    release_pending();
    drop_plans();
    if (check_.sched) ggml_backend_sched_free(check_.sched);
    if (mtp1_.sched) ggml_backend_sched_free(mtp1_.sched);
    if (sched_) ggml_backend_sched_free(sched_);
    free_kv();
    for (auto* b : wbufs_) if (b) ggml_backend_buffer_free(b);
    release_kept_ram(kept_ram_);
    if (wctx_) ggml_free(wctx_);
    for (auto* b : gpus_) if (b && b != cpu_) ggml_backend_free(b);
    if (cpu_) { forget_cpu_backend(cpu_); ggml_backend_free(cpu_); }
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
        has_layer0 = false;                              // any layer-0 tensor (Flash-Next's layers have no attn_norm)
        for (int64_t k = 0; k < gguf_get_n_tensors(g) && !has_layer0; ++k) has_layer0 = std::strncmp(gguf_get_tensor_name(g, k), "blk.0.", 6) == 0;
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

// Loads, then checks the VRAM left for the GPU's working memory against what this model, context and batch really
// need (measured: the compute buffers the scheduler reserved, plus room for the GPU engine's scratch memory and the
// driver). Too little: loads again with that much kept free, and notes it in opt.vram_note for next time, so it
// happens once per model and context size.
std::unique_ptr<Model> Model::load(const std::string& path, const LoadOptions& opt) {
    LoadOptions o = opt;
    // (per context, batch, connections and placement kind: the hot-expert split needs more than whole layers)
    // ("v2": 0.8.4 sizes the scratch to the card; a note from 0.8.3 held 640 MB of scratch on every card, so isn't read)
    const std::string key = "v2 " + std::to_string(opt.max_ctx) + " " + std::to_string(opt.batch) + " " + std::to_string(opt.parallel) + " " +
                            opt.split_mode + (opt.expert_usage.empty() ? "" : "+hot") + (opt.live_hot ? "+live" : "") + " ";
    if (o.vram_reserve_mb <= 0 && !o.vram_note.empty()) {
        std::ifstream f(std::filesystem::u8path(o.vram_note));
        for (std::string line; std::getline(f, line); )
            if (line.compare(0, key.size(), key) == 0) o.vram_reserve_mb = std::atoi(line.c_str() + key.size());
    }
    auto m = load_once(path, o);
    const int used = m->place_.vram_reserve_mb;
    // Scratch beyond the measured compute buffers (the GPU engine's own scratch memory, the driver): a share of the
    // card, 192 to 640 MB. Measured on an RTX 3070 Laptop (8 GB) with Qwen3.6-35B-A3B: 400 MB kept free in all (100
    // of it scratch) never touched shared memory over a 1,500-token reply, with 1 GB of the card still free, and fit
    // 2,865 VRAM experts instead of 2,459 with 936 MB kept (40.0 against 38.7 tok/s). 640 on a 20 GB+ card, as before.
    size_t vram_total = 0;
    if (!m->gpus_.empty()) { size_t fr = 0; ggml_backend_dev_memory(ggml_backend_get_device(m->gpus_[0]), &fr, &vram_total); }
    const int runtime = [&]{ const char* e = std::getenv("TIERLM_VRAM_SCRATCH_MB");
                             return e ? std::atoi(e) : std::clamp((int)(vram_total >> 20) / 32, 192, 640); }();
    const int need = (int)(m->place_.compute_gpu >> 20) + runtime;
    const bool tight = m->place_.compute_gpu > 0 && need > used + 64 && opt.vram_budget_mb <= 0 && !std::getenv("TIERLM_VRAM_BYTES");
    if (!o.vram_note.empty() && m->place_.compute_gpu > 0 && need > o.vram_reserve_mb) {   // remember it (the most seen: no reload back and forth)
        std::vector<std::string> keep;
        { std::ifstream f(std::filesystem::u8path(o.vram_note)); for (std::string line; std::getline(f, line); ) if (line.compare(0, key.size(), key) != 0 && !line.empty()) keep.push_back(line); }
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::u8path(o.vram_note).parent_path(), ec);
        std::ofstream f(std::filesystem::u8path(o.vram_note), std::ios::trunc);
        for (const auto& l : keep) f << l << "\n";
        f << key << std::max(need, 512) << "\n";
    }
    if (!tight) return m;
    std::fprintf(stderr, "VRAM: this model needs %d MB free for the GPU's working memory and %d MB was kept; loading again "
                         "with %d MB kept, so replies don't slow down when VRAM runs out (once: it's remembered)\n", need, used, need);
    m.reset();
    o.vram_reserve_mb = need;
    return load_once(path, o);
}

std::unique_ptr<Model> Model::load_once(const std::string& path, const LoadOptions& opt) {
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
    // Experts split between VRAM and RAM: the batch size from which the GPU does the RAM experts' work too (they are
    // copied over for the batch, every batch). ggml's own rule is 32 tokens. Measured on the RTX 5090 + Ryzen 9
    // 9950X3D with Flash-Next (28 GB of experts in RAM, the VRAM experts serving 45-80% of the picks), prompt
    // tokens/s with the CPU doing them against the GPU, batches of up to 1,024: 39 tokens 133-156 against 22-28,
    // 121 tokens 120-141 against 70, 285 tokens 367 against 182, 505 tokens 347-498 against 307, 772 tokens 305
    // against 386, 1,031 tokens 484 against 490. With the usual batches of 256, a 1,031-token prompt: 323-490 against
    // 154-168. So from 512 tokens on a 24 GB card or more (reached only with a bigger batch size set); elsewhere
    // (not measured yet) from 128, below which a copy of every RAM expert can't pay. TIERLM_COLD_OFFLOAD_MIN: another size.
    if (const char* e = std::getenv("TIERLM_COLD_OFFLOAD_MIN")) m->cold_offload_min_ = std::max(1, std::atoi(e));
    else if (m->main_ != m->cpu_) {
        size_t vfree = 0, vtotal = 0;
        m->device_memory(vfree, vtotal);
        m->cold_offload_min_ = vtotal >= ((size_t)23 << 30) && !m->integrated_gpu() ? 512 : 128;
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
        ggml_init_params wp{ (2 * n_total + 256) * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };   // + an MTP file, hot-expert parts
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
    // A whole number per layer: the key holds a list (Nemotron), or one number for every layer; missing: zeros.
    auto per_layer = [&](const std::string& k, int n) -> std::vector<int> {
        std::vector<int> v((size_t)n, 0);
        const int64_t id = gguf_find_key(g, k.c_str());
        if (id < 0) return v;
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) { std::fill(v.begin(), v.end(), (int)gguf_get_val_u32(g, id)); return v; }
        const auto t = gguf_get_arr_type(g, id);
        const void* d = gguf_get_arr_data(g, id);
        const int cnt = (int)gguf_get_arr_n(g, id);
        for (int i = 0; i < n && i < cnt; ++i)
            v[i] = t == GGUF_TYPE_INT32 || t == GGUF_TYPE_UINT32 ? (int)((const int32_t*)d)[i]
                 : t == GGUF_TYPE_INT64 || t == GGUF_TYPE_UINT64 ? (int)((const int64_t*)d)[i]
                 : t == GGUF_TYPE_INT16 || t == GGUF_TYPE_UINT16 ? (int)((const uint16_t*)d)[i]
                 : t == GGUF_TYPE_INT8 || t == GGUF_TYPE_UINT8 ? (int)((const uint8_t*)d)[i] : 0;
        return v;
    };
    // (a key that holds a list gives its largest entry)
    auto u32 = [&](const std::string& k, bool required) -> int {
        int64_t id = gguf_find_key(g, k.c_str());
        if (id < 0) { if (required) throw std::runtime_error("missing key " + k); return 0; }
        if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
            int mx = 0;
            for (int v : per_layer(k, (int)gguf_get_arr_n(g, id))) mx = std::max(mx, v);
            return mx;
        }
        return (int)gguf_get_val_u32(g, id);
    };
    auto f32 = [&](const std::string& k, float def) -> float {
        int64_t id = gguf_find_key(g, k.c_str());
        return id < 0 ? def : gguf_get_val_f32(g, id);
    };

    const std::string arch = str("general.architecture");
    if (!arch_supported(arch)) { gguf_free(g); throw std::runtime_error("the " + arch + " architecture is not supported yet"); }
    // PrismML's Bonsai 2: weights stored rotated (a Hadamard transform, declared under prism.hadamard.*). Without the
    // matching transform on the way in, the model loads and writes gibberish, so it is refused until tierlm has it.
    for (int64_t i = 0; i < gguf_get_n_kv(g); ++i)
        if (std::strncmp(gguf_get_key(g, i), "prism.hadamard", 14) == 0) {
            gguf_free(g);
            throw std::runtime_error("Bonsai 2 (ternary weights stored in a rotated form) is not supported yet");
        }
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
        else if (type == "yarn" && factor > 1.0f) {
            // YaRN, exactly as Hugging Face computes it (transformers' _compute_yarn_parameters): each rotation
            // frequency is blended between its original value and value / factor along a ramp between two
            // "correction" dimensions; cos and sin are scaled by 0.1 ln(factor) + 1. ggml's own YaRN rounds the
            // ramp's ends to whole dimensions; gpt-oss is trained without that rounding ("truncate": false), so the
            // frequencies are computed here and given to ggml as per-dimension factors instead.
            const int dim = hp.n_rot > 0 ? hp.n_rot : (hp.head_dim > 0 ? hp.head_dim : hp.n_embd / hp.n_head);
            const double base = hp.rope_base, orig = u32(arch + ".rope.scaling.original_context_length", false);
            const double beta_fast = f32(arch + ".rope.scaling.yarn_beta_fast", 32.0f), beta_slow = f32(arch + ".rope.scaling.yarn_beta_slow", 1.0f);
            const bool truncate = arch != "gpt-oss";
            if (orig > 0 && dim > 0) {
                auto corr = [&](double rot) { return dim * std::log(orig / (rot * 2 * 3.14159265358979323846)) / (2 * std::log(base)); };
                double lo = corr(beta_fast), hi = corr(beta_slow);
                if (truncate) { lo = std::floor(lo); hi = std::ceil(hi); }
                lo = std::max(lo, 0.0); hi = std::min(hi, (double)dim - 1);
                if (lo == hi) hi += 0.001;
                hp.rope_ff.resize(dim / 2);
                for (int i = 0; i < dim / 2; ++i) {
                    const double ramp = std::min(1.0, std::max(0.0, (i - lo) / (hi - lo)));
                    const double e = 1.0 - ramp;                         // share of the original (extrapolated) frequency
                    hp.rope_ff[i] = (float)(1.0 / ((1.0 - e) / factor + e));
                }
                hp.rope_attn_factor = (float)(0.1 * std::log((double)factor) + 1.0);
            }
        }
        else if (!type.empty() && type != "none" && factor > 1.0f)
            std::fprintf(stderr, "warning: rope scaling '%s' is not supported yet; contexts beyond the original length will degrade\n", type.c_str());
    }
    // The variations
    hp.rope_neox     = arch != "llama";
    hp.qk_norm       = arch == "olmoe" ? Hparams::kFull : (arch == "qwen3" || arch == "qwen3moe" || arch == "flint") ? Hparams::kPerHead : Hparams::kNone;
    hp.qkv_bias      = arch == "qwen2" || arch == "gpt-oss";
    hp.expert_renorm = arch == "qwen3moe" || arch == "qwen35moe" || arch == "qwen3next" || arch == "qwen4exp" || arch == "llama";   // OLMoE doesn't renormalise
    hp.expert_scale  = f32(arch + ".expert_weights_scale", 0.0f);
    if (arch == "qwen3moe" || arch == "qwen35moe" || arch == "qwen3next" || arch == "qwen4exp" || arch == "flint") {
        int64_t id = gguf_find_key(g, (arch + ".expert_weights_norm").c_str());
        if (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_BOOL) hp.expert_renorm = gguf_get_val_bool(g, id);
    }
    if (arch == "gpt-oss") {
        // Router: pick the top experts by score, then softmax over just those. Every other layer (the even ones,
        // from the first) attends only to the last n_swa tokens, unless the file lists the layers itself.
        hp.expert_softmax_first = false;
        hp.expert_renorm = false;
        hp.swiglu_oai = true;
        hp.n_swa = u32(arch + ".attention.sliding_window", false);
        hp.swa.assign(hp.n_layer, false);
        const int64_t pid = gguf_find_key(g, (arch + ".attention.sliding_window_pattern").c_str());
        if (pid >= 0 && gguf_get_kv_type(g, pid) == GGUF_TYPE_ARRAY && gguf_get_arr_type(g, pid) == GGUF_TYPE_BOOL) {
            const bool* d = (const bool*)gguf_get_arr_data(g, pid);
            for (int i = 0; i < hp.n_layer && i < (int)gguf_get_arr_n(g, pid); ++i) hp.swa[i] = d[i];
        } else {
            const int pattern = pid >= 0 ? (int)u32(arch + ".attention.sliding_window_pattern", false) : 2;
            for (int i = 0; i < hp.n_layer; ++i) hp.swa[i] = pattern <= 0 || i % pattern < pattern - 1;
        }
    }
    if (arch == "flint") {
        // Flint (Recursive): experts scored by a sigmoid, the top ones' scores renormalised; the first layers dense;
        // sliding-window layers (with attention sinks, as gpt-oss) and full-attention layers, and only the sliding
        // ones rotate positions: the full-attention layers use none.
        hp.expert_sigmoid = true;
        if (gguf_find_key(g, (arch + ".expert_gating_func").c_str()) >= 0 && u32(arch + ".expert_gating_func", false) == 1)
            hp.expert_sigmoid = false;                                // (1 = softmax)
        hp.n_dense_lead = (int)u32(arch + ".leading_dense_block_count", false);
        hp.n_swa = u32(arch + ".attention.sliding_window", false);
        auto bools = [&](const std::string& key, std::vector<bool>& out) {
            const int64_t id = gguf_find_key(g, key.c_str());
            if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, id) != GGUF_TYPE_BOOL) return false;
            const bool* d = (const bool*)gguf_get_arr_data(g, id);
            out.assign(hp.n_layer, false);
            for (int i = 0; i < hp.n_layer && i < (int)gguf_get_arr_n(g, id); ++i) out[i] = d[i];
            return true;
        };
        if (!bools(arch + ".attention.sliding_window_pattern", hp.swa)) {   // Hugging Face's default layout
            hp.swa.assign(hp.n_layer, false);
            for (int i = 0; i < hp.n_layer; ++i) hp.swa[i] = !(i < 2 || i % 4 == 1 || i == hp.n_layer - 1);
        }
        if (!bools(arch + ".attention.rope_pattern", hp.rope_layer)) hp.rope_layer = hp.swa;
        if (hp.n_swa <= 0) hp.swa.assign(hp.n_layer, false);
    }
    // DeepSeek V2/V3, Kimi K2.x, Moonlight: multi-head latent attention (see Hparams::mla) and DeepSeek's experts.
    // The per-head key/value expansion is stored three ways in GGUF files: one attn_kv_b tensor ([kv_lora,
    // n_head*(nope+v_dim)], as Hugging Face has it), or attn_k_b and attn_v_b, either already in the layout used here
    // or transposed. Sizes come from the tensors themselves, which can't disagree with the data; whatever isn't in
    // the layout used here is rearranged once at load (mla_derive below).
    struct MlaDerive { int layer; ggml_tensor *kv = nullptr, *k = nullptr, *v = nullptr; bool k_t = false, v_t = false; ggml_tensor *kb = nullptr, *vb = nullptr; };
    std::vector<MlaDerive> mla_derive;
    std::set<const ggml_tensor*> skip_load;                        // in the file but not loaded (rearranged instead)
    if (arch == "deepseek2") {
        hp.mla = true;
        hp.rope_neox = false;                                     // rotated in adjacent pairs (Hugging Face's "interleave")
        hp.q_lora  = (int)u32(arch + ".attention.q_lora_rank", false);
        hp.kv_lora = (int)u32(arch + ".attention.kv_lora_rank", true);
        hp.n_rot   = (int)u32(arch + ".rope.dimension_count", true);
        const int kl = gguf_find_key(g, (arch + ".attention.key_length_mla").c_str()) >= 0 ? (int)u32(arch + ".attention.key_length_mla", false) : (int)u32(arch + ".attention.key_length", false);
        const int vl = gguf_find_key(g, (arch + ".attention.value_length_mla").c_str()) >= 0 ? (int)u32(arch + ".attention.value_length_mla", false) : (int)u32(arch + ".attention.value_length", false);
        hp.nope = kl - hp.n_rot; hp.v_dim = vl;
        const int H = hp.n_head, R = hp.kv_lora;
        for (int il = 0; il < hp.n_layer; ++il) {
            const std::string b = "blk." + std::to_string(il) + ".";
            MlaDerive d; d.layer = il;
            d.kv = ggml_get_tensor(m->wctx_, (b + "attn_kv_b.weight").c_str());
            d.k  = ggml_get_tensor(m->wctx_, (b + "attn_k_b.weight").c_str());
            d.v  = ggml_get_tensor(m->wctx_, (b + "attn_v_b.weight").c_str());
            if (d.k && d.v) {
                const int64_t kn = ggml_nelements(d.k), vn = ggml_nelements(d.v);
                if (kn % ((int64_t)H * R) || vn % ((int64_t)H * R)) { gguf_free(g); throw std::runtime_error("unexpected attn_k_b/attn_v_b sizes in " + b); }
                hp.nope = (int)(kn / ((int64_t)H * R)); hp.v_dim = (int)(vn / ((int64_t)H * R));
                d.k_t = d.k->ne[0] != hp.nope;                     // stored with kv_lora first: rearrange
                d.v_t = d.v->ne[0] != R;
                if (d.kv) skip_load.insert(d.kv);                  // (both kinds in one file: the split ones are used)
                if (!d.k_t && !d.v_t) continue;                    // used as they are
            } else if (d.kv) {
                if (d.kv->ne[0] != R || d.kv->ne[1] % H) { gguf_free(g); throw std::runtime_error("unexpected attn_kv_b shape in " + b); }
                if (hp.nope <= 0 || hp.v_dim <= 0 || d.kv->ne[1] / H != hp.nope + hp.v_dim) {   // trust the tensor over the keys
                    const int per = (int)(d.kv->ne[1] / H);
                    if (hp.v_dim > 0 && per > hp.v_dim) hp.nope = per - hp.v_dim; else { gguf_free(g); throw std::runtime_error("can't tell this model's key and value sizes apart"); }
                }
                d.k = d.v = nullptr;
            } else { gguf_free(g); throw std::runtime_error("missing attn_kv_b (or attn_k_b and attn_v_b) in " + b); }
            // rearranged copies, kept in f32 when the file is f32 (the tests) and f16 otherwise
            const ggml_type src_t = (d.kv ? d.kv : d.k)->type, ty = src_t == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16;
            if (d.kv || d.k_t) { d.kb = ggml_new_tensor_3d(m->wctx_, ty, hp.nope, R, H); ggml_set_name(d.kb, (b + "attn_k_b.tierlm").c_str()); }
            if (d.kv || d.v_t) { d.vb = ggml_new_tensor_3d(m->wctx_, ty, R, hp.v_dim, H); ggml_set_name(d.vb, (b + "attn_v_b.tierlm").c_str()); }
            if (d.kv) skip_load.insert(d.kv);
            if (d.k_t) skip_load.insert(d.k);
            if (d.v_t) skip_load.insert(d.v);
            mla_derive.push_back(d);
        }
        hp.head_dim  = R + hp.n_rot;                               // the cache: one compressed vector per token...
        hp.n_head_kv = 1;                                          // ...shared by every head
        // Scale: 1/sqrt of the real per-head key size, times YaRN's mscale squared when the context is extended
        // (Hugging Face's yarn_get_mscale(factor, mscale_all_dim); the file keeps 0.1 * mscale_all_dim as
        // yarn_log_multiplier). cos/sin stay unscaled: DeepSeek's mscale and mscale_all_dim are equal.
        float mscale = 1.0f;
        const float factor = f32(arch + ".rope.scaling.factor", 1.0f);
        if (str((arch + ".rope.scaling.type").c_str()) == "yarn" && factor > 1.0f)
            mscale = 1.0f + f32(arch + ".rope.scaling.yarn_log_multiplier", 0.1f) * std::log(factor);
        hp.kq_scale = mscale * mscale / std::sqrt((float)(hp.nope + hp.n_rot));
        hp.rope_attn_factor = 1.0f;
        // experts: the first layers dense; a shared expert; softmax (V2) or sigmoid with a bias for choosing (V3)
        hp.n_dense_lead  = (int)u32(arch + ".leading_dense_block_count", false);
        hp.expert_sigmoid = gguf_find_key(g, (arch + ".expert_gating_func").c_str()) >= 0 && u32(arch + ".expert_gating_func", false) == 2;
        {
            const int64_t id = gguf_find_key(g, (arch + ".expert_weights_norm").c_str());
            hp.expert_renorm = id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_BOOL && gguf_get_val_bool(g, id);
        }
        hp.n_group    = (int)u32(arch + ".expert_group_count", false);
        hp.topk_group = (int)u32(arch + ".expert_group_used_count", false);
        hp.group_sum2 = hp.expert_sigmoid;
        if (hp.n_group <= 1 || hp.topk_group <= 0 || hp.topk_group >= hp.n_group || hp.n_expert % std::max(1, hp.n_group)) hp.n_group = hp.topk_group = 0;
    }
    // Qwen3.8 Flash-Next: Qwen3.5's DeltaNet and gated-attention layers and experts, plus hyper-connections, the
    // per-layer n-gram embedding and sparse attention (see Hparams::hc, ple_layer, qsa_budget).
    if (arch == "qwen4exp") {
        hp.hc = (int)u32(arch + ".hyper_connection.count", true);
        hp.ple_layer.assign(hp.n_layer, false);
        const int64_t pid = gguf_find_key(g, (arch + ".ple.layers").c_str());
        if (pid >= 0 && gguf_get_kv_type(g, pid) == GGUF_TYPE_ARRAY)
            for (size_t i = 0; i < gguf_get_arr_n(g, pid); ++i) {
                const int l = ((const int32_t*)gguf_get_arr_data(g, pid))[i];
                if (l >= 0 && l < hp.n_layer) hp.ple_layer[l] = true;
            }
        if (hp.any_ple()) {
            auto u64s = [&](const std::string& k) {
                std::vector<int64_t> v;
                const int64_t id = gguf_find_key(g, k.c_str());
                if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return v;
                const auto t = gguf_get_arr_type(g, id);
                for (size_t i = 0; i < gguf_get_arr_n(g, id); ++i)
                    v.push_back(t == GGUF_TYPE_UINT64 || t == GGUF_TYPE_INT64 ? ((const int64_t*)gguf_get_arr_data(g, id))[i]
                                                                              : (int64_t)((const int32_t*)gguf_get_arr_data(g, id))[i]);
                return v;
            };
            const int ngram = (int)u32(arch + ".ple.ngram_size", true), hpn = (int)u32(arch + ".ple.heads_per_ngram", true);
            const int32_t eos = (int32_t)u32(arch + ".ple.eos_token_id", true);
            auto mult = u64s(arch + ".ple.layer_multipliers"), size = u64s(arch + ".ple.head_vocab_sizes"), off = u64s(arch + ".ple.head_offsets");
            if ((int)mult.size() != ngram || (int)size.size() != (ngram - 1) * hpn || size.size() != off.size()) {
                gguf_free(g); throw std::runtime_error("this Flash-Next file's n-gram settings are incomplete");
            }
            hp.ngram.set(ngram, hpn, eos, mult, size, off);
            hp.ple_dim = (int)u32(arch + ".embedding_length_per_layer_input", true);
            hp.ple_kernel = (int)u32(arch + ".ple.conv_kernel", false);
            if (hp.ple_kernel <= 0) hp.ple_kernel = 4;
        }
        hp.qsa_budget = (int)u32(arch + ".attention.indexer.top_k", false);
        hp.qsa_heads  = (int)u32(arch + ".attention.indexer.head_count", false);
        hp.qsa_dim    = (int)u32(arch + ".attention.indexer.key_length", false);
        hp.qsa_ratio.assign(hp.n_layer, 0);
        if (const int64_t rid = gguf_find_key(g, (arch + ".attention.compress_ratios").c_str()); rid >= 0 && gguf_get_kv_type(g, rid) == GGUF_TYPE_ARRAY)
            for (int i = 0; i < hp.n_layer && i < (int)gguf_get_arr_n(g, rid); ++i) hp.qsa_ratio[i] = ((const int32_t*)gguf_get_arr_data(g, rid))[i];
        if (gguf_find_key(g, (arch + ".ple.image_token_id").c_str()) >= 0) hp.ple_image_tok = (int32_t)u32(arch + ".ple.image_token_id", false);
    }
    if (arch == "nemotron_h_moe") {
        // Which block is which isn't written down: a block with no key/value heads and no feed-forward width is a
        // Mamba-2 mixer, one with heads is attention, the rest are experts.
        hp.mamba = hp.hybrid = true;
        const std::vector<int> kvh = per_layer(arch + ".attention.head_count_kv", hp.n_layer);
        const std::vector<int> ff = per_layer(arch + ".feed_forward_length", hp.n_layer);
        hp.n_expert_used_l = per_layer(arch + ".expert_used_count", hp.n_layer);
        hp.block.assign(hp.n_layer, Hparams::kFfn);
        hp.recurrent.assign(hp.n_layer, false);
        hp.n_expert_l.assign(hp.n_layer, 0);
        for (int i = 0; i < hp.n_layer; ++i) {
            if (kvh[i] > 0) { hp.block[i] = Hparams::kAttn; if (kvh[i] != hp.n_head_kv) { gguf_free(g); throw std::runtime_error("attention blocks with different head counts are not supported yet"); } }
            else if (ff[i] == 0) { hp.block[i] = Hparams::kMamba; hp.recurrent[i] = true; }
            else {
                const ggml_tensor* r = ggml_get_tensor(m->wctx_, ("blk." + std::to_string(i) + ".ffn_gate_inp.weight").c_str());
                if (!r) { gguf_free(g); throw std::runtime_error("Nemotron blocks without experts are not supported yet"); }
                hp.n_expert_l[i] = (int)r->ne[1];
                if (hp.n_expert_used_l[i] <= 0 || hp.n_expert_used_l[i] > hp.n_expert_l[i]) { gguf_free(g); throw std::runtime_error("unexpected experts-per-token count"); }
            }
        }
        hp.ssm_conv = u32(arch + ".ssm.conv_kernel", true);
        hp.m_state  = u32(arch + ".ssm.state_size", true);
        hp.m_groups = u32(arch + ".ssm.group_count", true);
        hp.m_inner  = u32(arch + ".ssm.inner_size", true);
        hp.m_heads  = u32(arch + ".ssm.time_step_rank", true);
        if (hp.m_heads <= 0 || hp.m_groups <= 0 || hp.m_inner % hp.m_heads || hp.m_heads % hp.m_groups || hp.m_inner % hp.m_groups) {
            gguf_free(g); throw std::runtime_error("unexpected Mamba sizes");
        }
        hp.latent = u32(arch + ".moe_latent_size", false);
        hp.rope_layer.assign(hp.n_layer, false);                  // no position encoding: the Mamba blocks carry the order
        hp.expert_sigmoid = true;                                 // each expert scored on its own, chosen with a bias,
        hp.expert_softmax_first = true;                           // (no softmax over the chosen ones)
        hp.expert_relu2 = true;
        const int64_t id = gguf_find_key(g, (arch + ".expert_weights_norm").c_str());
        hp.expert_renorm = id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_BOOL && gguf_get_val_bool(g, id);
    }
    if (arch == "qwen35" || arch == "qwen35moe" || arch == "qwen3next" || arch == "qwen4exp") {
        hp.hybrid = true;
        hp.gated_attn = true;
        // Qwen3-Next (Qwen3-Coder-Next) files keep the linear-attention value heads grouped by key head, as
        // Hugging Face stores them; Qwen3.5's converter reorders them so ggml can broadcast the key heads.
        hp.v_grouped = arch == "qwen3next";
        // Qwen3.8 Flash-Next gates DeltaNet's output with sigmoid(z), where Qwen3.5 uses silu(z). The file doesn't say:
        // it is the released model's output_gate_type ("sigmoid"; Hugging Face's code defaults to silu when unset).
        hp.dn_sigmoid_gate = arch == "qwen4exp";
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
        // Qwen3.5 reads images too (with its vision file): 3-part positions, sections from the file.
        const int64_t sid = gguf_find_key(g, (arch + ".rope.dimension_sections").c_str());
        if (arch != "qwen3next" && sid >= 0 && gguf_get_kv_type(g, sid) == GGUF_TYPE_ARRAY &&
            (gguf_get_arr_type(g, sid) == GGUF_TYPE_INT32 || gguf_get_arr_type(g, sid) == GGUF_TYPE_UINT32)) {
            const int32_t* d = (const int32_t*)gguf_get_arr_data(g, sid);
            for (int i = 0; i < 4 && i < (int)gguf_get_arr_n(g, sid); ++i) hp.mrope_sections[i] = d[i];
            hp.mrope = hp.mrope_sections[0] + hp.mrope_sections[1] + hp.mrope_sections[2] > 0;
        } else if (arch != "qwen3next" && hp.n_rot == 64) {          // files without the key: Qwen3.5's own sizes
            hp.mrope_sections[0] = 11; hp.mrope_sections[1] = 11; hp.mrope_sections[2] = 10;
            hp.mrope = true;
        }
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
    // TIERLM_KV_F32=1); q8_0 halves the memory again with practically no quality loss, q5_0 takes a third,
    // q4_0 a little over a quarter.
    // Quantized caches are read by flash attention, so they turn it on, and need a head size that is a whole
    // number of 32-value blocks.
    {
        std::string want = opt.kv_type;
        if (want.empty()) {
            if (const char* e = std::getenv("TIERLM_KV_TYPE")) want = e;
            else want = (opt.kv_f32 >= 0 ? opt.kv_f32 == 1 : std::getenv("TIERLM_KV_F32") != nullptr) ? "f32" : "f16";
        }
        m->kv_t_ = want == "f32" ? GGML_TYPE_F32 : want == "q8_0" ? GGML_TYPE_Q8_0 : want == "q5_0" ? GGML_TYPE_Q5_0 :
                   want == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
        if (ggml_is_quantized((ggml_type)m->kv_t_) && hp.head_dim % 32 != 0) {
            std::fprintf(stderr, "note: this model's head size (%d) can't use a %s cache; using f16\n", hp.head_dim, want.c_str());
            m->kv_t_ = GGML_TYPE_F16;
        }
        m->kv_f32_ = m->kv_t_ == GGML_TYPE_F32;
    }
    // Conversation memory per layer: a KV cache for attention layers, a small fixed-size state for DeltaNet layers.
    auto layer_mem = [&](int il) -> size_t {
        if (hp.is_recurrent(il))
            return ((size_t)(hp.ssm_conv - 1) * hp.ssm_conv_dim() + (size_t)hp.ssm_mem_elems()) * 4 *
                   (size_t)std::max(1, opt.parallel);
        if (hp.block_kind(il) == Hparams::kFfn) return 0;          // Nemotron's expert blocks keep nothing
        // keys and values; MLA keeps one compressed vector per token, which is both
        return (size_t)(hp.mla ? 1 : 2) * ggml_row_size((ggml_type)m->kv_t_, (int64_t)hp.head_dim * hp.n_head_kv) * (size_t)m->max_ctx_;
    };
    size_t kv_need = 0;
    for (int il = 0; il < nl_all; ++il) kv_need += layer_mem(il);
    const bool tied = ggml_get_tensor(m->wctx_, "output.weight") == nullptr;
    const int G = (int)m->gpus_.size();                  // GPUs in use (0: CPU only)
    ggml_backend_buffer_type_t gpu_buft = G ? ggml_backend_get_default_buffer_type(m->main_) : nullptr;
    ggml_backend_buffer_type_t cpu_buft = cpu_buffer_type();
    std::vector<ggml_backend_buffer_type_t> bufts(G);
    std::vector<size_t> budgets(G, 0);                     // VRAM each GPU can give to weights and the KV cache
    // Integrated graphics (Intel Arc / Iris in a laptop, AMD Radeon in an APU) have no memory of their own: their
    // "VRAM" is borrowed from system RAM, and the driver reports up to about half of RAM as available. Counting that
    // as VRAM and then giving the rest of the model the same RAM again overcommitted memory (a laptop with Intel Arc
    // graphics locked up). So an integrated GPU gets at most 40% of the RAM that's free, and what it takes comes off
    // the RAM budget below.
    size_t igpu_taken = 0;
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
            const bool igpu = ggml_backend_dev_type(ggml_backend_get_device(m->gpus_[d])) == GGML_BACKEND_DEVICE_TYPE_IGPU;
            if (igpu) {
                size_t fr = 0, tr = 0;
                if (ggml_backend_dev_t cd = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) ggml_backend_dev_memory(cd, &fr, &tr);
                if (fr) avail = std::min(avail, (size_t)(fr * 0.4));
            }
            if (opt.vram_budget_mb > 0) avail = std::min(avail, (size_t)opt.vram_budget_mb << 20);
            const size_t reserve = (size_t)(opt.vram_reserve_mb > 0 ? opt.vram_reserve_mb : 512) << 20;   // compute memory, scratch, driver (measured after: see load)
            m->place_.vram_reserve_mb = (int)(reserve >> 20);
            budgets[d] = avail > reserve ? avail - reserve : 0;
            if (!forced.empty()) budgets[d] = forced[std::min((size_t)d, forced.size() - 1)];
            if (igpu) igpu_taken += budgets[d] + reserve;
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
        if (skip_load.count(t)) continue;                                // rearranged at load instead (MLA)
        const std::string name = ggml_get_name(t);
        int prio = 1, layer = -1;
        bool own = false;                                                // a group of its own
        if (name == "token_embd.weight") { prio = tied ? 1 : 9; own = true; }   // only looked up by row: fine in RAM
        else if (name == "per_layer_token_embd.weight") { prio = 9; own = true; }   // Flash-Next's n-gram table: 16 rows a token
        else if (name == "output.weight") { prio = 1; layer = hp.n_layer; own = true; }
        else if (name.compare(0, 4, "blk.") == 0) {
            layer = std::atoi(name.c_str() + 4);
            if (layer >= nl_all) continue;                               // draft (MTP) blocks: only when speculating
            const std::string rest = name.substr(name.find('.', 4) + 1);
            if (by_layer) prio = 2;                                      // the whole layer moves together
            else if (rest.find("_exps") != std::string::npos)
                // The draft block's experts go into VRAM before the model's: every guess and every catch-up runs
                // them, and left in RAM (where they landed whenever the model's experts filled VRAM) each of those
                // calls went GPU - CPU - GPU. TIERLM_MTP_VRAM=0: after the model's, as before 0.8.7.
                prio = layer == hp.n_layer && !(std::getenv("TIERLM_MTP_VRAM") && !std::strcmp(std::getenv("TIERLM_MTP_VRAM"), "0")) ? 3 : 4;
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
    // Hot experts (experts in RAM, one GPU, with usage counts from earlier use): once the attention, router and KV
    // cache are placed, the VRAM left goes to the most-used individual experts rather than to whole expert layers.
    // A layer whose chosen experts are only some of its experts is split in two (see Layer::map_hot).
    struct HotSplit { ggml_tensor *orig, *hot, *cold; };
    // (bias: the experts' biases, gpt-oss. They are small, so they aren't split: the file's tensor stays in RAM for
    // the RAM part and a whole copy goes into VRAM for the VRAM part, both looked up with the router's own ids.)
    struct HotLayer { std::vector<int> ids; std::vector<HotSplit> parts; std::vector<std::pair<ggml_tensor*, ggml_tensor*>> bias;
                      ggml_tensor *map_hot = nullptr, *map_cold = nullptr, *mask = nullptr;
                      bool cold_mapped = false; };
    std::map<int, HotLayer> hot_layers;
    const int E = hp.n_expert;
    std::vector<float> usage_env;                                     // tests: TIERLM_EXPERT_USAGE=<seed> makes up usage counts
    if (const char* e = std::getenv("TIERLM_EXPERT_USAGE"); e && opt.expert_usage.empty() && hp.is_moe()) {
        uint32_t x = (uint32_t)std::strtoul(e, nullptr, 10) * 2654435761u + 1;
        usage_env.resize((size_t)hp.n_layer * E);
        for (auto& u : usage_env) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; u = (float)(x % 1000) * (x % 3 == 0 ? 0.0f : 1.0f); }
    }
    const std::vector<float>& usage = opt.expert_usage.empty() ? usage_env : opt.expert_usage;
    hp.expert_bias = ggml_get_tensor(m->wctx_, "blk.0.ffn_down_exps.bias") != nullptr;
    auto is_bias = [](const ggml_tensor* t) { const std::string nm = ggml_get_name(t); return nm.size() > 5 && nm.compare(nm.size() - 5, 5, ".bias") == 0; };
    // (Not yet for Nemotron, whose blocks have different numbers of experts: whole expert layers there.)
    const bool use_hot = G == 1 && !by_layer && hp.is_moe() && !hp.mamba && usage.size() == (size_t)hp.n_layer * E;
    bool live = use_hot && (opt.live_hot || std::getenv("TIERLM_LIVE_EXPERTS"));
    std::vector<std::vector<int>> hot_ids(hp.n_layer);
    bool hot_decided = false;
    auto choose_hot = [&]() {
        std::vector<size_t> per(hp.n_layer, 0);                      // VRAM per expert, per layer
        for (const Group& gr : groups)
            if (gr.prio == 4 && gr.layer >= 0 && gr.layer < hp.n_layer) {
                size_t bias = 0;                                         // (their VRAM copies are whole: taken off the budget first)
                for (const ggml_tensor* t : gr.t)
                    if (is_bias(t)) bias += GGML_PAD(ggml_backend_buft_get_alloc_size(gpu_buft, t), ggml_backend_buft_get_alignment(gpu_buft));
                per[gr.layer] = (gr.bytes - bias) / E + 1;
                budget -= std::min(budget, bias);
            }
        struct Cand { float u; int l, e; };
        std::vector<Cand> cs;
        for (int l = 0; l < hp.n_layer; ++l)
            for (int e = 0; e < E; ++e)
                if (per[l] && usage[(size_t)l * E + e] > 0) cs.push_back({usage[(size_t)l * E + e], l, e});
        std::stable_sort(cs.begin(), cs.end(), [](const Cand& a, const Cand& b) { return a.u > b.u; });
        std::vector<std::vector<char>> taken(hp.n_layer, std::vector<char>(E, 0));
        auto take = [&](int l, int e) {
            const size_t need = per[l] * (hot_ids[l].empty() ? 2 : 1);   // the first one also pays for the zero expert
            if (need > budget) return false;
            budget -= need;
            hot_ids[l].push_back(e);
            taken[l][e] = 1;
            return true;
        };
        for (const Cand& c : cs) take(c.l, c.e);
        // VRAM still free: the never-used experts too, layer by layer (whole layers need no split)
        for (int l = 0; l < hp.n_layer; ++l)
            for (int e = 0; e < E && per[l]; ++e)
                if (!taken[l][e]) take(l, e);
        // Both parts of a split layer need at least n_used experts (the zero expert included), so each token's
        // picks can be made distinct there (see distinct_ids). Too few in VRAM: the layer stays in RAM; too few
        // left in RAM: the least-used chosen ones go back (they were taken busiest first).
        const int need_part = std::max(1, hp.n_expert_used);
        for (int l = 0; l < hp.n_layer; ++l) {
            int H = (int)hot_ids[l].size();
            if (H == 0 || H == E) continue;
            if (H + 1 < need_part) { budget += per[l] * (H + 1); hot_ids[l].clear(); continue; }
            while (E - H + 1 < need_part) { hot_ids[l].pop_back(); budget += per[l]; --H; }
        }
        for (int l = 0; l < hp.n_layer; ++l) {
            if ((int)hot_ids[l].size() == E) budget += per[l];        // complete layer: no zero expert needed
            std::sort(hot_ids[l].begin(), hot_ids[l].end());
        }
    };
    if (G <= 1) {
        bool kv_decided = false, layers_full = false;
        for (const Group& gr : groups) {
            if (!by_layer && gr.prio >= 3 && !kv_decided) {                  // the KV cache goes before the FFNs
                const bool fit = gpu_buft && kv_need <= budget;
                if (fit) budget -= kv_need;
                std::fill(kv_gpu.begin(), kv_gpu.end(), fit);
                kv_decided = true;
            }
            if (use_hot && gr.prio == 4 && gr.layer >= 0 && gr.layer < hp.n_layer) {
                if (!hot_decided) {
                    choose_hot();
                    hot_decided = true;
                    // Live: an expert moving into VRAM trades places with the one it replaces, which goes into
                    // its RAM slot, so the RAM part stays the size of the fixed set's (no second copy).
                }
                const int H = (int)hot_ids[gr.layer].size();
                if (H == E || H == 0) { for (ggml_tensor* t : gr.t) (H ? on_gpu : on_cpu).push_back(t); continue; }
                HotLayer& hl = hot_layers[gr.layer];
                hl.ids = hot_ids[gr.layer];
                for (ggml_tensor* t : gr.t) {
                    const std::string nm = ggml_get_name(t);
                    if (is_bias(t)) {
                        ggml_tensor* tv = ggml_dup_tensor(m->wctx_, t);
                        ggml_set_name(tv, (nm + ".vram").c_str());
                        on_gpu.push_back(tv);
                        on_cpu.push_back(t);
                        hl.bias.push_back({t, tv});
                        continue;
                    }
                    ggml_tensor* th = ggml_new_tensor_3d(m->wctx_, t->type, t->ne[0], t->ne[1], H + 1);
                    ggml_tensor* tc = ggml_new_tensor_3d(m->wctx_, t->type, t->ne[0], t->ne[1], E - H + 1);
                    ggml_set_name(th, (nm + ".hot").c_str());
                    ggml_set_name(tc, (nm + ".cold").c_str());
                    on_gpu.push_back(th);
                    on_cpu.push_back(tc);
                    hl.parts.push_back({t, th, tc});
                }
                const std::string b = "blk." + std::to_string(gr.layer) + ".tier_";
                hl.map_hot = ggml_new_tensor_2d(m->wctx_, GGML_TYPE_I32, 1, E);  ggml_set_name(hl.map_hot, (b + "map_hot").c_str());
                hl.map_cold = ggml_new_tensor_2d(m->wctx_, GGML_TYPE_I32, 1, E); ggml_set_name(hl.map_cold, (b + "map_cold").c_str());
                hl.mask = ggml_new_tensor_2d(m->wctx_, GGML_TYPE_F32, 1, E);    ggml_set_name(hl.mask, (b + "mask").c_str());
                on_cpu.push_back(hl.map_hot); on_cpu.push_back(hl.map_cold); on_cpu.push_back(hl.mask);
                continue;
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
    // ---- the SSD tier: what doesn't fit in RAM is read from a drive as it's used ----
    // The weights left for RAM are checked against the RAM there is (less a margin for Windows and the other
    // programs). If they don't fit, whole groups of them are memory-mapped from a file instead of loaded: the
    // least-used expert layers first (each token reads only the experts it picks, and the operating system keeps the
    // recently read ones in spare RAM), then, for models without experts or still too big, the first layers. A layer
    // whose experts are split for the hot-expert placement keeps its RAM part in RAM (that part isn't laid out as
    // in the file). TIERLM_SSD_LAYERS=n (tests): exactly n groups go to the drive, whatever fits.
    std::vector<ggml_tensor*> on_ssd;
    std::vector<ggml_tensor*> cold_inplace;                 // split layers' RAM parts read from the model file (RAM short)
    {
        std::map<const ggml_tensor*, const Source*> src_of;
        for (const Source& s : sources) src_of[s.t] = &s;
        size_t ram_need = 0;
        for (ggml_tensor* t : on_cpu) ram_need += ggml_backend_buft_get_alloc_size(cpu_buft, t);
        size_t free_ram = 0, total_ram = 0;
        if (ggml_backend_dev_t cd = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) ggml_backend_dev_memory(cd, &free_ram, &total_ram);
        free_ram = free_ram > igpu_taken ? free_ram - igpu_taken : 0;   // what integrated graphics borrow is RAM too
        size_t ram_budget = 0;
        if (opt.ram_budget_mb > 0) ram_budget = (size_t)opt.ram_budget_mb << 20;
        else if (free_ram) {
            const size_t margin = std::max((size_t)3 << 29, (size_t)(total_ram * 0.12));   // 1.5 GB, or 12% of RAM
            // the KV cache and compute memory of the layers on the CPU come out of RAM too
            size_t kv_ram = 0;
            for (int il = 0; il < nl_all; ++il) if (!kv_gpu[il]) kv_ram += layer_mem(il);
            const size_t other = margin + kv_ram + ((size_t)1 << 30);
            ram_budget = free_ram > other ? free_ram - other : 0;
        }
        int forced = -1;
        if (const char* e = std::getenv("TIERLM_SSD_LAYERS")) forced = std::atoi(e);
        const char* off_env = std::getenv("TIERLM_SSD");             // TIERLM_SSD=0: never (tests)
        const int force_cold = std::getenv("TIERLM_COLD_MAPPED") ? std::atoi(std::getenv("TIERLM_COLD_MAPPED")) : -1;   // tests
        const bool want = opt.ssd != 0 && !(off_env && std::string(off_env) == "0") && (forced > 0 || force_cold > 0 || (opt.ssd == 1 && ram_budget && ram_need > ram_budget) ||
                                           (opt.ssd < 0 && ram_budget && ram_need > ram_budget));
        if (want) {
            // Groups that can go: per layer, its expert tensors (MoE), else all of its weights; in the file's layout.
            struct Cand { int layer; bool experts; float use; std::vector<ggml_tensor*> t; size_t bytes = 0; };
            std::map<std::pair<int, int>, Cand> cand;                 // (0 experts / 1 other, layer)
            std::vector<Cand> first;                                  // tables looked up by row: the drive suits them best
            for (ggml_tensor* t : on_cpu) {
                const std::string n = ggml_get_name(t);
                if (n == "per_layer_token_embd.weight" && src_of.count(t)) {
                    Cand c; c.layer = -1; c.experts = true; c.use = -1; c.t.push_back(t); c.bytes = ggml_nbytes(t);
                    first.push_back(c);
                    continue;
                }
                if (n.compare(0, 4, "blk.") != 0 || n.find("tier_") != std::string::npos || !src_of.count(t)) continue;
                if (n.size() > 5 && (n.compare(n.size() - 5, 5, ".cold") == 0 || n.compare(n.size() - 4, 4, ".hot") == 0)) continue;
                const int l = std::atoi(n.c_str() + 4);
                const bool ex = n.find("_exps") != std::string::npos;
                Cand& c = cand[{ex ? 0 : 1, l}];
                c.layer = l; c.experts = ex;
                c.t.push_back(t);
                c.bytes += ggml_nbytes(t);
            }
            std::vector<Cand> order;
            for (auto& [k, c] : cand) {
                c.use = 0;
                if (c.experts && usage.size() == (size_t)hp.n_layer * E && c.layer < hp.n_layer)
                    for (int e = 0; e < E; ++e) c.use += usage[(size_t)c.layer * E + e];
                order.push_back(c);
            }
            std::stable_sort(order.begin(), order.end(), [](const Cand& a, const Cand& b) {
                if (a.experts != b.experts) return a.experts;            // expert layers first
                if (a.use != b.use) return a.use < b.use;                // the least used first
                return a.layer < b.layer;
            });
            order.insert(order.begin(), first.begin(), first.end());
            std::set<const ggml_tensor*> moved;
            std::set<int> ssd_layer_set;
            int taken = 0;
            for (const Cand& c : order) {
                // Tables looked up by row (the n-gram table) go to the drive once RAM is short; expert layers only when
                // they can't fit at all (more than the PC's RAM less 3 GB): Windows makes room for weights a little over
                // the budget above, and running them from RAM is much faster (see the RAM parts of split layers below).
                const size_t relaxed = total_ram > ((size_t)3 << 30) ? total_ram - ((size_t)3 << 30) : 0;
                if (forced >= 0 ? taken >= forced : ram_need <= (c.layer < 0 ? ram_budget : std::max(ram_budget, relaxed))) break;
                for (ggml_tensor* t : c.t) { moved.insert(t); on_ssd.push_back(t); ram_need -= std::min(ram_need, ggml_backend_buft_get_alloc_size(cpu_buft, t)); }
                if (c.layer >= 0) ssd_layer_set.insert(c.layer); else m->place_.ssd_ngram = true;
                ++taken;
            }
            if (!moved.empty()) {
                std::vector<ggml_tensor*> keep;
                for (ggml_tensor* t : on_cpu) if (!moved.count(t)) keep.push_back(t);
                on_cpu.swap(keep);
            }
            // Still too much, and the layers are split between VRAM and RAM (their RAM parts are made at load, so they
            // aren't in the file to read from): the least-used layers' RAM parts become the file's own expert tensors,
            // read where they are, as used. The picks still go to the experts in VRAM first; the ones in RAM are read
            // from the file, which Windows keeps in spare RAM and can drop when short (without writing it to its page
            // file, as it would the RAM part's memory: that is what slowed a 64 GB PC down to a crawl).
            // Only when it can't fit at all (more than the PC's RAM less 3 GB): measured on a 64 GB PC with Flash-Next, RAM parts
            // 10 GB over the usual budget still ran at 48 tok/s held in RAM (Windows makes room), against 31 read from the file.
            const bool cannot_fit = total_ram && ram_need + ((size_t)3 << 30) > total_ram;
            if (((forced < 0 && cannot_fit && force_cold != 0) || force_cold > 0) && !hot_layers.empty()) {   // (TIERLM_COLD_MAPPED=0: never)
                struct Cold { float use; int layer; size_t bytes; };
                std::vector<Cold> cs;
                for (auto& [l, hl] : hot_layers) {
                    bool ok = true;
                    size_t bytes = 0;
                    for (const HotSplit& sp : hl.parts) {
                        auto it = src_of.find(sp.orig);
                        if (it == src_of.end() || it->second->offset % 32 != 0) ok = false;
                        bytes += ggml_backend_buft_get_alloc_size(cpu_buft, sp.cold);
                    }
                    if (!ok) continue;
                    float use = 0;
                    std::vector<char> hot(E, 0);
                    for (int e : hl.ids) hot[e] = 1;
                    for (int e = 0; e < E; ++e) if (!hot[e]) use += usage[(size_t)l * E + e];
                    cs.push_back({use, l, bytes});
                }
                std::stable_sort(cs.begin(), cs.end(), [](const Cold& a, const Cold& b) { return a.use < b.use; });
                std::set<const ggml_tensor*> gone;
                for (const Cold& c : cs) {
                    if (force_cold > 0 ? m->place_.cold_mapped_layers >= force_cold : ram_need <= ram_budget) break;
                    HotLayer& hl = hot_layers[c.layer];
                    for (HotSplit& sp : hl.parts) { gone.insert(sp.cold); cold_inplace.push_back(sp.orig); sp.cold = sp.orig; }
                    hl.cold_mapped = true;
                    ram_need -= std::min(ram_need, c.bytes);
                    ++m->place_.cold_mapped_layers;
                }
                if (!gone.empty()) {
                    std::vector<ggml_tensor*> keep;
                    for (ggml_tensor* t : on_cpu) if (!gone.count(t)) keep.push_back(t);
                    on_cpu.swap(keep);
                    std::fprintf(stderr, "RAM: the experts in RAM of %d split layers are read from the model's file as used "
                                 "(RAM is too small to hold them all)\n", m->place_.cold_mapped_layers);
                }
            }
            m->place_.ssd_layers = (int)ssd_layer_set.size();
            if (forced < 0 && ram_need > ram_budget)
                std::fprintf(stderr, "note: the weights left for RAM (%.1f GB) are still more than it has room for (%.1f GB)\n",
                             ram_need / 1e9, ram_budget / 1e9);
        }
        if (!on_cpu.empty()) {
            m->place_.ram_need = ram_need + ((size_t)1 << 30);
            m->place_.ram_free = free_ram;
            if (const char* e = std::getenv("TIERLM_TEST_RAM_FREE_MB")) m->place_.ram_free = (uint64_t)std::atoll(e) << 20;   // tests: the warning
        }
    }
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
                                    n.find("conv1d") == std::string::npos && n.find("tier_") == std::string::npos &&
                                    // live hot experts copy experts out of the RAM part into VRAM, which needs the
                                    // file's layout (a repacked tensor can't be read back)
                                    !(live && n.size() > 5 && n.compare(n.size() - 5, 5, ".cold") == 0);
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
    // The SSD tier's weights: mapped from the model's own file where it is (ssd_dir empty), or from tierlm's copy of
    // just those weights in ssd_dir (written now if it isn't there or doesn't match this model).
    std::set<const ggml_tensor*> ssd_set(on_ssd.begin(), on_ssd.end());
    size_t copy_total = 0;                                   // bytes copied to the drive now (progress)
    if (!on_ssd.empty()) {
        std::map<const ggml_tensor*, const Source*> src_of;
        for (const Source& s : sources) src_of[s.t] = &s;
        std::string ssd_dir = opt.ssd_dir;
        if (const char* e = std::getenv("TIERLM_SSD_DIR"); e && ssd_dir.empty()) ssd_dir = e;   // tests
        bool in_place = ssd_dir.empty();
        for (ggml_tensor* t : on_ssd) if (src_of.at(t)->offset % 32 != 0) in_place = false;   // ggml needs aligned data
        auto map_into = [&](const std::string& file, const std::vector<std::pair<ggml_tensor*, size_t>>& at) {
            auto mf = std::make_unique<MappedFile>();
            mf->open(file);
            ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr((void*)mf->data(), mf->size());
            if (!b) throw std::runtime_error("cannot use " + file + " for the SSD tier");
            ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (auto [t, off] : at) {
                if (off + ggml_nbytes(t) > mf->size()) throw std::runtime_error(file + " is shorter than expected");
                ggml_backend_tensor_alloc(b, t, (char*)mf->data() + off);
                m->place_.ssd_weights += ggml_nbytes(t);
            }
            m->wbufs_.push_back(b);
            m->maps_.push_back(std::move(mf));
            m->place_.ssd_file = file;
        };
        try {
            if (in_place) {
                std::map<int, std::vector<std::pair<ggml_tensor*, size_t>>> by_part;
                for (ggml_tensor* t : on_ssd) by_part[src_of.at(t)->part].push_back({t, src_of.at(t)->offset});
                for (auto& [part, at] : by_part) map_into(paths.at(part), at);
            } else {
                // The copies: a text header naming each tensor, then the tensors, each at a 4 KB boundary. Tables looked
                // up by row (Flash-Next's n-gram table, 35 GB) get a file of their own, written once; the expert layers
                // another, named after which layers it holds, so trying other settings (Find the fastest settings)
                // doesn't rewrite tens of GB each time. The last few layer sets are kept.
                const std::string dir = ssd_dir.empty() ? std::filesystem::u8path(path).parent_path().u8string() : ssd_dir;
                const std::string stem = std::filesystem::u8path(path).stem().u8string();
                unsigned long long model_bytes = 0;
                for (const auto& pp : paths) { std::error_code ec; model_bytes += std::filesystem::file_size(std::filesystem::u8path(pp), ec); }
                std::vector<ggml_tensor*> tables, layers;
                for (ggml_tensor* t : on_ssd) (std::string(ggml_get_name(t)) == "per_layer_token_embd.weight" ? tables : layers).push_back(t);
                size_t want_all = 0;
                for (ggml_tensor* t : on_ssd) want_all += ggml_nbytes(t);
                bool wrote = false;
                auto use_copy = [&](const std::vector<ggml_tensor*>& ts, bool is_tables) {
                    std::string head = "tierlm-ssd 1\nmodel " + std::filesystem::u8path(path).filename().u8string() + " " + std::to_string(model_bytes) + "\n";
                    for (ggml_tensor* t : ts) head += std::string(ggml_get_name(t)) + " " + std::to_string(ggml_nbytes(t)) + "\n";
                    head += "end\n";
                    uint64_t hsh = 1469598103934665603ull;                     // FNV-1a of the header: which tensors
                    for (unsigned char ch : head) { hsh ^= ch; hsh *= 1099511628211ull; }
                    char tag[24];
                    std::snprintf(tag, sizeof tag, "%08llx", (unsigned long long)(hsh & 0xffffffffull));
                    const std::string file = (std::filesystem::u8path(dir) / (stem + (is_tables ? ".tables" : std::string(".") + tag) + ".tierlm-ssd")).u8string();
                    std::vector<std::pair<ggml_tensor*, size_t>> at;
                    size_t off = GGML_PAD(head.size(), 4096);
                    for (ggml_tensor* t : ts) { at.push_back({t, off}); off = GGML_PAD(off + ggml_nbytes(t), 4096); }
                    const size_t want_size = off;
                    bool ok = false;
                    {
                        std::error_code ec;
                        if (std::filesystem::file_size(std::filesystem::u8path(file), ec) == want_size && !ec) {
                            std::ifstream f(std::filesystem::u8path(file), std::ios::binary);
                            std::string got(head.size(), '\0');
                            ok = f.read(got.data(), (std::streamsize)got.size()) && got == head;
                        }
                    }
                    if (!ok) {
                        std::error_code ec;
                        std::filesystem::create_directories(std::filesystem::u8path(dir), ec);
                        // room: the old single-file copy (tierlm 0.8.2) and the oldest layer sets go first
                        std::filesystem::remove(std::filesystem::u8path(dir) / (stem + ".tierlm-ssd"), ec);
                        if (!is_tables) {
                            std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> olds;
                            for (const auto& de : std::filesystem::directory_iterator(std::filesystem::u8path(dir), ec)) {
                                const std::string fn = de.path().filename().u8string();
                                if (fn.size() > stem.size() + 12 && fn.compare(0, stem.size() + 1, stem + ".") == 0 &&
                                    fn.find(".tables.") == std::string::npos && fn.size() >= 11 && fn.compare(fn.size() - 11, 11, ".tierlm-ssd") == 0)
                                    olds.push_back({de.last_write_time(ec), de.path()});
                            }
                            std::sort(olds.begin(), olds.end());
                            while (olds.size() > 2) { std::filesystem::remove(olds.front().second, ec); olds.erase(olds.begin()); }
                        }
                        unsigned long long fr = 0, tot = 0;
                        if (drive_space(dir, fr, tot) && fr < want_size + ((unsigned long long)1 << 30))
                            throw std::runtime_error("the SSD tier needs " + std::to_string((want_size + (1ull << 30) - 1) >> 30) + " GB free on the drive of " +
                                                     dir + "; it has " + std::to_string(fr >> 30) + " GB");
                        std::fprintf(stderr, "SSD tier: copying %.1f GB of weights to %s (once)\n", want_size / 1e9, file.c_str());
                        const std::string tmp = file + ".part";
                        {
                            std::ofstream o(std::filesystem::u8path(tmp), std::ios::binary | std::ios::trunc);
                            if (!o) throw std::runtime_error("cannot write " + tmp);
                            std::vector<char> pad(4096, 0), buf((size_t)64 << 20);
                            o.write(head.data(), (std::streamsize)head.size());
                            size_t pos = head.size();
                            std::vector<std::ifstream> in;
                            for (const auto& pp : paths) in.emplace_back(std::filesystem::u8path(pp), std::ios::binary);
                            for (auto [t, toff] : at) {
                                o.write(pad.data(), (std::streamsize)(toff - pos)); pos = toff;
                                const Source* sr = src_of.at(t);
                                auto& f = in.at(sr->part);
                                f.seekg((std::streamoff)sr->offset);
                                for (size_t left = ggml_nbytes(t); left; ) {
                                    const size_t n = std::min(left, buf.size());
                                    if (!f.read(buf.data(), (std::streamsize)n)) throw std::runtime_error("short read on " + paths.at(sr->part));
                                    o.write(buf.data(), (std::streamsize)n);
                                    left -= n; pos += n; copy_total += n;
                                    if (opt.progress) opt.progress(0.5f * (float)((double)copy_total / (double)std::max<size_t>(want_all, 1)));
                                }
                            }
                            o.write(pad.data(), (std::streamsize)(want_size - pos));
                            if (!o) throw std::runtime_error("writing " + tmp + " failed (drive full?)");
                        }
                        // On the drive now, not later: Windows would otherwise write it out over the next minutes,
                        // while the model runs (and is measured), slowing down everything that reads the drives.
                        flush_to_disk(tmp);
                        std::filesystem::rename(std::filesystem::u8path(tmp), std::filesystem::u8path(file), ec);
                        if (ec) throw std::runtime_error("cannot rename " + tmp + ": " + ec.message());
                        wrote = true;
                    }
                    map_into(file, at);
                };
                if (!tables.empty()) use_copy(tables, true);
                if (!layers.empty()) use_copy(layers, false);
                (void)wrote;
            }
        } catch (...) { gguf_free(g); throw; }
        std::fprintf(stderr, "SSD tier: %s%d layers, %.1f GB, read as needed from %s\n", m->place_.ssd_ngram ? "the n-gram table + " : "", m->place_.ssd_layers,
                     m->place_.ssd_weights / 1e9, m->place_.ssd_file.c_str());
    }
    m->weights_bytes_ = m->place_.gpu_weights + m->place_.cpu_weights + m->place_.ssd_weights;
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
            m->place_.attn_gpu += in_vram("attn_norm.weight") || in_vram("hc_attn_norm.weight");   // (Flash-Next: its mixer's norm)
            const bool f = hp.n_expert > 0 ? in_vram("ffn_down_exps.weight") : in_vram("ffn_up.weight");
            m->place_.ffn_gpu += f;
            m->place_.ffn_vram.push_back(f);
        }
        if (use_hot) {
            m->place_.hot.assign(hp.n_layer, {});
            for (int i = 0; i < hp.n_layer; ++i) {
                m->place_.hot[i] = hot_ids[i];
                m->place_.hot_experts += (int)hot_ids[i].size();
            }
        }
    }

    if (!cold_inplace.empty()) {                            // (see "RAM: ..." above) mapped from the model's own file
        std::map<const ggml_tensor*, const Source*> src_of;
        for (const Source& s : sources) src_of[s.t] = &s;
        std::map<int, std::vector<std::pair<ggml_tensor*, size_t>>> by_part;
        for (ggml_tensor* t : cold_inplace) by_part[src_of.at(t)->part].push_back({t, src_of.at(t)->offset});
        try {
            for (auto& [part, at] : by_part) {
                auto mf = std::make_unique<MappedFile>();
                mf->open(paths.at(part));
                ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr((void*)mf->data(), mf->size());
                if (!b) throw std::runtime_error("cannot map " + paths.at(part));
                ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                for (auto [t, off] : at) {
                    if (off + ggml_nbytes(t) > mf->size()) throw std::runtime_error(paths.at(part) + " is shorter than expected");
                    ggml_backend_tensor_alloc(b, t, (char*)mf->data() + off);
                }
                m->wbufs_.push_back(b);
                m->maps_.push_back(std::move(mf));
            }
        } catch (...) { gguf_free(g); throw; }
    }
    // ---- stream every tensor's data from its file into wherever the tier loader put it ----
    std::vector<std::ifstream> files;
    for (const auto& p : paths) {
        files.emplace_back(std::filesystem::u8path(p), std::ios::binary);   // UTF-8 path, also on Windows
        if (!files.back()) { gguf_free(g); throw std::runtime_error("cannot open " + p); }
    }
    std::vector<char> chunk;
    std::map<const ggml_tensor*, std::pair<const HotLayer*, const HotSplit*>> split_of;
    std::map<const ggml_tensor*, ggml_tensor*> bias_copy;
    for (const auto& [l, hl] : hot_layers) {
        for (const HotSplit& sp : hl.parts) split_of[sp.orig] = {&hl, &sp};
        for (const auto& [orig, vram] : hl.bias) bias_copy[orig] = vram;
        // Each expert's place in the VRAM part and the RAM part; picks that live in the other part go to the zero expert.
        const int H = (int)hl.ids.size();
        std::vector<int32_t> mh(E, H), mc(E, E - H);
        std::vector<float> mk(E, 0.0f);
        int ci = 0;
        std::vector<char> is_hot(E, 0);
        for (int k = 0; k < H; ++k) { mh[hl.ids[k]] = k; mk[hl.ids[k]] = 1.0f; is_hot[hl.ids[k]] = 1; }
        for (int e = 0; e < E; ++e) if (!is_hot[e]) mc[e] = ci++;
        // read from the file: every expert at its own index, and no zero expert (E: none, see skip_zero_ids)
        if (hl.cold_mapped) for (int e = 0; e < E; ++e) mc[e] = is_hot[e] ? E : e;
        ggml_backend_tensor_set(hl.map_hot, mh.data(), 0, ggml_nbytes(hl.map_hot));
        ggml_backend_tensor_set(hl.map_cold, mc.data(), 0, ggml_nbytes(hl.map_cold));
        ggml_backend_tensor_set(hl.mask, mk.data(), 0, ggml_nbytes(hl.mask));
    }
    size_t total_bytes = 0, done_bytes = 0, next_report = 0;
    for (const Source& s : sources) if ((s.t->buffer && !ssd_set.count(s.t)) || split_of.count(s.t)) total_bytes += ggml_nbytes(s.t);
    // after a copy to the drive: the copy was the first half of the progress bar, reading the rest the second
    const float pbase = copy_total ? 0.5f : 0.0f, pscale = copy_total ? 0.5f : 1.0f;
    auto report = [&](float f) { if (opt.progress) opt.progress(pbase + pscale * f); };
    double read_s = 0, split_s = 0, set_s = 0;                         // (TIERLM_LOAD_TRACE: where loading's time goes)
    using lclock = std::chrono::steady_clock;
    auto since = [](lclock::time_point t) { return std::chrono::duration<double>(lclock::now() - t).count(); };
    for (const Source& s : sources) {
        if (ssd_set.count(s.t)) continue;                              // read from the drive when used
        if (auto it = split_of.find(s.t); it != split_of.end()) {      // an expert tensor split into VRAM and RAM parts
            const HotLayer& hl = *it->second.first;
            const HotSplit& sp = *it->second.second;
            const size_t n = ggml_nbytes(s.t), per = s.t->nb[2];
            done_bytes += n;
            chunk.resize(n);
            auto& f = files[s.part];
            const auto t_read = lclock::now();
            f.seekg((std::streamoff)s.offset);
            if (!f.read(chunk.data(), (std::streamsize)n)) { gguf_free(g); throw std::runtime_error("short read on " + paths[s.part]); }
            read_s += since(t_read);
            // The VRAM part and the RAM part, each put together in a buffer kept from tensor to tensor (not made and
            // zeroed again for each: for gpt-oss-120b that took longer than reading the file), then set in one go.
            // The last expert of each part is the zero expert.
            static std::vector<char> hb, cb;
            const size_t hn = ggml_nbytes(sp.hot), cn = ggml_nbytes(sp.cold);
            const auto t_split = lclock::now();
            if (hb.size() < hn) hb.resize(hn);
            if (!hl.cold_mapped && cb.size() < cn) cb.resize(cn);
            std::vector<char> is_hot(E, 0);
            for (int k = 0; k < (int)hl.ids.size(); ++k) { std::memcpy(hb.data() + (size_t)k * per, chunk.data() + (size_t)hl.ids[k] * per, per); is_hot[hl.ids[k]] = 1; }
            if (hn > hl.ids.size() * per) std::memset(hb.data() + hl.ids.size() * per, 0, hn - hl.ids.size() * per);
            ggml_backend_tensor_set(sp.hot, hb.data(), 0, hn);
            if (!hl.cold_mapped) {                                             // (mapped: the file is it)
                size_t ci = 0;
                for (int e = 0; e < E; ++e)
                    if (!is_hot[e]) std::memcpy(cb.data() + ci++ * per, chunk.data() + (size_t)e * per, per);
                if (cn > ci * per) std::memset(cb.data() + ci * per, 0, cn - ci * per);
                ggml_backend_tensor_set(sp.cold, cb.data(), 0, cn);
            }
            split_s += since(t_split);
            report(total_bytes ? (float)((double)done_bytes / total_bytes) : 1.0f);
            continue;
        }
        if (!s.t->buffer) continue;                                    // not loaded (draft blocks)
        const size_t n = ggml_nbytes(s.t);
        done_bytes += n;
        if (opt.progress && (done_bytes >= next_report || done_bytes == total_bytes)) {
            report(total_bytes ? (float)((double)done_bytes / total_bytes) : 1.0f);
            next_report = done_bytes + total_bytes / 200;
        }
        chunk.resize(n);
        auto& f = files[s.part];
        const auto t_read = lclock::now();
        f.seekg((std::streamoff)s.offset);
        if (!f.read(chunk.data(), (std::streamsize)n)) { gguf_free(g); throw std::runtime_error("short read on " + paths[s.part]); }
        read_s += since(t_read);
        const auto t_set = lclock::now();
        ggml_backend_tensor_set(s.t, chunk.data(), 0, n);
        if (auto bc = bias_copy.find(s.t); bc != bias_copy.end()) ggml_backend_tensor_set(bc->second, chunk.data(), 0, n);
        set_s += since(t_set);
    }
    if (std::getenv("TIERLM_LOAD_TRACE"))
        std::fprintf(stderr, "load: reading the file %.1f s, splitting experts between VRAM and RAM %.1f s, copying the rest %.1f s\n",
                     read_s, split_s, set_s);
    // MLA: the per-head key/value expansion rearranged into the layout attention uses (see mla_derive above): read
    // from the file, turned into numbers (whatever the file's type), reordered, and stored.
    if (!mla_derive.empty()) {
        std::map<const ggml_tensor*, const Source*> src_of;
        for (const Source& s : sources) src_of[s.t] = &s;
        auto read_f32 = [&](const ggml_tensor* t) {
            const Source* sr = src_of.at(t);
            std::vector<char> raw(ggml_nbytes(t));
            auto& f = files[sr->part];
            f.seekg((std::streamoff)sr->offset);
            if (!f.read(raw.data(), (std::streamsize)raw.size())) throw std::runtime_error("short read on " + paths[sr->part]);
            std::vector<float> out((size_t)ggml_nelements(t));
            if (t->type == GGML_TYPE_F32) std::memcpy(out.data(), raw.data(), raw.size());
            else {
                const auto* tr = ggml_get_type_traits(t->type);
                if (!tr || !tr->to_float) throw std::runtime_error(std::string("can't read ") + ggml_get_name(t) + " (" + ggml_type_name(t->type) + ")");
                tr->to_float(raw.data(), out.data(), (int64_t)out.size());
            }
            return out;
        };
        auto store = [&](ggml_tensor* dst, const std::vector<float>& v) {
            if (dst->type == GGML_TYPE_F32) { ggml_backend_tensor_set(dst, v.data(), 0, v.size() * 4); return; }
            std::vector<ggml_fp16_t> h(v.size());
            ggml_fp32_to_fp16_row(v.data(), h.data(), (int64_t)v.size());
            ggml_backend_tensor_set(dst, h.data(), 0, h.size() * 2);
        };
        const int H = hp.n_head, R = hp.kv_lora, N = hp.nope, Vd = hp.v_dim;
        try {
            for (const MlaDerive& d : mla_derive) {
                std::vector<float> kb, vb;
                if (d.kv) {                                      // rows: per head, N key rows then Vd value rows, each R long
                    const std::vector<float> a = read_f32(d.kv);
                    if (d.kb) { kb.resize((size_t)H * R * N);
                        for (int h = 0; h < H; ++h) for (int n = 0; n < N; ++n) for (int r = 0; r < R; ++r)
                            kb[((size_t)h * R + r) * N + n] = a[((size_t)h * (N + Vd) + n) * R + r]; }
                    if (d.vb) { vb.resize((size_t)H * Vd * R);
                        for (int h = 0; h < H; ++h) for (int v = 0; v < Vd; ++v)
                            std::memcpy(&vb[((size_t)h * Vd + v) * R], &a[((size_t)h * (N + Vd) + N + v) * R], sizeof(float) * R); }
                } else {
                    if (d.kb) {                                  // stored [R, N, H]: per head, N rows of R
                        const std::vector<float> a = read_f32(d.k);
                        kb.resize(a.size());
                        for (int h = 0; h < H; ++h) for (int n = 0; n < N; ++n) for (int r = 0; r < R; ++r)
                            kb[((size_t)h * R + r) * N + n] = a[((size_t)h * N + n) * R + r];
                    }
                    if (d.vb) {                                  // stored [Vd, R, H]: per head, R rows of Vd
                        const std::vector<float> a = read_f32(d.v);
                        vb.resize(a.size());
                        for (int h = 0; h < H; ++h) for (int r = 0; r < R; ++r) for (int v = 0; v < Vd; ++v)
                            vb[((size_t)h * Vd + v) * R + r] = a[((size_t)h * R + r) * Vd + v];
                    }
                }
                if (d.kb) store(d.kb, kb);
                if (d.vb) store(d.vb, vb);
            }
        } catch (...) { gguf_free(g); throw; }
    }
    gguf_free(g);

    auto T = [&](const std::string& name, bool required = true) -> ggml_tensor* {
        ggml_tensor* t = ggml_get_tensor(m->wctx_, name.c_str());
        if (!t && required) throw std::runtime_error("missing tensor " + name);
        return t;
    };
    m->tok_embd_   = T("token_embd.weight");
    m->out_norm_   = T("output_norm.weight", hp.hc == 0);             // (Flash-Next: its last stream mixer instead)
    if (hp.hc) {
        m->out_hc_norm_ = T("output_hc_norm.weight");
        m->out_hc_down_ = T("output_hc_down.weight");
        m->out_hc_up_   = T("output_hc_up.weight");
        if (hp.any_ple()) m->ple_embd_ = T("per_layer_token_embd.weight");
    }
    m->output_     = T("output.weight", false);
    m->rope_freqs_ = T("rope_freqs.weight", false);   // Llama 3.1+ long-context frequency factors
    if (!m->output_) m->output_ = m->tok_embd_;      // tied embeddings
    hp.n_vocab = (int)m->tok_embd_->ne[1];

    auto bind_hot = [&](Layer& L, int i) {                         // a layer whose experts are split into VRAM and RAM parts
        auto it = hot_layers.find(i);
        if (it == hot_layers.end()) return;
        for (const HotSplit& sp : it->second.parts) {
            const std::string nm = ggml_get_name(sp.orig);
            if (nm.find("ffn_gate_up_exps") != std::string::npos) { L.gate_up_xh = sp.hot; L.gate_up_xc = sp.cold; }
            else if (nm.find("ffn_gate_exps") != std::string::npos) { L.gate_xh = sp.hot; L.gate_xc = sp.cold; }
            else if (nm.find("ffn_up_exps") != std::string::npos) { L.up_xh = sp.hot; L.up_xc = sp.cold; }
            else if (nm.find("ffn_down_exps") != std::string::npos) { L.down_xh = sp.hot; L.down_xc = sp.cold; }
        }
        for (const auto& [orig, vram] : it->second.bias) {
            const std::string nm = ggml_get_name(orig);
            if (nm.find("ffn_gate_exps") != std::string::npos) L.gate_xbh = vram;
            else if (nm.find("ffn_up_exps") != std::string::npos) L.up_xbh = vram;
            else if (nm.find("ffn_down_exps") != std::string::npos) L.down_xbh = vram;
        }
        L.map_hot = it->second.map_hot; L.map_cold = it->second.map_cold; L.hot_mask = it->second.mask;
        L.cold_mapped = it->second.cold_mapped;
        if (live && !it->second.cold_mapped) {                        // the starting set; adapt_hot moves experts from here
            LiveLayer ll;
            ll.layer = i;
            ll.H = (int)it->second.ids.size();
            ll.slot_of.assign((size_t)std::max(E, 0), -1);
            ll.cold_of.assign((size_t)std::max(E, 0), -1);
            ll.in_slot = it->second.ids;
            for (int k = 0; k < ll.H; ++k) ll.slot_of[ll.in_slot[k]] = k;
            for (int e = 0, ci = 0; e < E; ++e) if (ll.slot_of[e] < 0) ll.cold_of[e] = ci++;   // (as load lays the RAM part out)
            // Learned use as the starting score, scaled so the busiest expert counts as a few recent picks: the
            // conversation takes over within a few dozen tokens.
            float mx = 0;
            for (int e = 0; e < E; ++e) mx = std::max(mx, usage[(size_t)i * E + e]);
            ll.score.assign((size_t)std::max(E, 0), 0.0f);
            for (int e = 0; e < E; ++e) ll.score[e] = mx > 0 ? 3.0f * usage[(size_t)i * E + e] / mx : 0.0f;
            for (const HotSplit& sp : it->second.parts) ll.parts.push_back({sp.hot, sp.cold});
            ll.map_hot = it->second.map_hot; ll.map_cold = it->second.map_cold; ll.mask = it->second.mask;
            m->live_.push_back(std::move(ll));
        }
    };
    auto bind_layer = [&](int i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        Layer L;
        L.index = i;
        L.attn_norm = T(b + "attn_norm.weight", hp.hc == 0);         // (Flash-Next: its stream mixers' norms)
        if (hp.hc) {
            L.hc_a_norm = T(b + "hc_attn_norm.weight"); L.hc_a_down = T(b + "hc_attn_down.weight");
            L.hc_a_up   = T(b + "hc_attn_up.weight");   L.hc_a_inj  = T(b + "hc_attn_inject.weight");
            L.hc_f_norm = T(b + "hc_ffn_norm.weight");  L.hc_f_down = T(b + "hc_ffn_down.weight");
            L.hc_f_up   = T(b + "hc_ffn_up.weight");    L.hc_f_inj  = T(b + "hc_ffn_inject.weight");
            if (i < (int)hp.ple_layer.size() && hp.ple_layer[i]) {
                L.ple_key = T(b + "ple_key.weight");             L.ple_value = T(b + "ple_value.weight");
                L.ple_norm_key = T(b + "ple_norm_key.weight");   L.ple_norm_query = T(b + "ple_norm_query.weight");
                L.ple_norm_conv = T(b + "ple_norm_conv.weight"); L.ple_conv = T(b + "ple_conv1d.weight");
            }
        }
        L.recurrent = hp.is_recurrent(i);
        if (hp.mamba) {                                               // Nemotron: one of three kinds of block
            L.kind = hp.block_kind(i);
            L.rope = false;
            if (L.kind == Hparams::kMamba) {
                L.ssm_in = T(b + "ssm_in.weight");          L.ssm_in_s = T(b + "ssm_in.scale", false);
                L.ssm_conv1d = T(b + "ssm_conv1d.weight");  L.ssm_conv_b = T(b + "ssm_conv1d.bias", false);
                L.ssm_dt = T(b + "ssm_dt.bias");            L.ssm_a = T(b + "ssm_a");
                L.ssm_d = T(b + "ssm_d");                   L.ssm_norm = T(b + "ssm_norm.weight", false);
                L.ssm_out = T(b + "ssm_out.weight");        L.ssm_out_s = T(b + "ssm_out.scale", false);
            } else if (L.kind == Hparams::kAttn) {
                L.wq = T(b + "attn_q.weight"); L.wk = T(b + "attn_k.weight"); L.wv = T(b + "attn_v.weight");
                L.wo = T(b + "attn_output.weight"); L.bo = T(b + "attn_output.bias", false);
            } else {
                L.router = T(b + "ffn_gate_inp.weight");    L.exp_probs_b = T(b + "exp_probs_b.bias", false);
                L.lat_down = T(b + "ffn_latent_down.weight", false);
                L.lat_up = T(b + "ffn_latent_up.weight", false);
                L.up_x = T(b + "ffn_up_exps.weight");       L.down_x = T(b + "ffn_down_exps.weight");
                L.up_s = T(b + "ffn_up_shexp.weight", false);
                if (L.up_s) L.down_s = T(b + "ffn_down_shexp.weight");
            }
            return L;
        }
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
        } else if (hp.mla) {                                          // DeepSeek / Kimi: latent attention
            if (hp.q_lora > 0) {
                L.wq_a     = T(b + "attn_q_a.weight");
                L.q_a_norm = T(b + "attn_q_a_norm.weight");
                L.wq_b     = T(b + "attn_q_b.weight");
            } else L.wq = T(b + "attn_q.weight");
            L.wkv_a     = T(b + "attn_kv_a_mqa.weight");
            L.kv_a_norm = T(b + "attn_kv_a_norm.weight");
            L.wk_b = ggml_get_tensor(m->wctx_, (b + "attn_k_b.tierlm").c_str());
            if (!L.wk_b) L.wk_b = T(b + "attn_k_b.weight");
            L.wv_b = ggml_get_tensor(m->wctx_, (b + "attn_v_b.tierlm").c_str());
            if (!L.wv_b) L.wv_b = T(b + "attn_v_b.weight");
            L.wo = T(b + "attn_output.weight");
        } else {
            if (hp.qsa_budget > 0 && i < (int)hp.qsa_ratio.size() && hp.qsa_ratio[i] > 0) {
                L.idx_q = T(b + "indexer.q_proj.weight");      L.idx_k = T(b + "indexer.k_proj.weight");
                L.idx_q_norm = T(b + "indexer.q_norm.weight"); L.idx_k_norm = T(b + "indexer.k_norm.weight");
            }
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
            L.bo    = T(b + "attn_output.bias", false);
            L.sinks = T(b + "attn_sinks.weight", false);
            L.swa   = hp.is_swa(i);
        }
        L.ffn_norm  = T(b + "ffn_norm.weight", false);                 // called post_attention_norm in some families
        if (!L.ffn_norm) L.ffn_norm = T(b + "post_attention_norm.weight", hp.hc == 0);
        if (i < (int)hp.rope_layer.size()) L.rope = hp.rope_layer[i];
        if (hp.is_moe() && i >= hp.n_dense_lead) {
            L.exp_probs_b = T(b + "exp_probs_b.bias", false);           // DeepSeek V3: bias for choosing only
            L.router = T(b + "ffn_gate_inp.weight");
            L.gate_up_x = T(b + "ffn_gate_up_exps.weight", false);      // gate and up stored together (newer files)
            if (!L.gate_up_x) {
                L.gate_x = T(b + "ffn_gate_exps.weight", false);
                if (!L.gate_x) throw std::runtime_error("this file stores experts as separate tensors (an old Mixtral GGUF); re-download a newer conversion");
                L.up_x   = T(b + "ffn_up_exps.weight");
            }
            L.down_x = T(b + "ffn_down_exps.weight");
            L.router_b = T(b + "ffn_gate_inp.bias", false);               // gpt-oss: biases on the router and experts
            L.gate_xb  = T(b + "ffn_gate_exps.bias", false);
            L.up_xb    = T(b + "ffn_up_exps.bias", false);
            L.down_xb  = T(b + "ffn_down_exps.bias", false);
            // Shared expert: runs on every token, scaled by its own sigmoid gate (Qwen3.5/3.6 MoE) or added in full (Flint).
            L.up_s = T(b + "ffn_up_shexp.weight", false);
            if (L.up_s) {
                L.gate_s     = T(b + "ffn_gate_shexp.weight");
                L.down_s     = T(b + "ffn_down_shexp.weight");
                L.gate_inp_s = T(b + "ffn_gate_inp_shexp.weight", false);
            }
        } else {
            L.gate = T(b + "ffn_gate.weight");
            L.up   = T(b + "ffn_up.weight");
            L.down = T(b + "ffn_down.weight");
        }
        return L;
    };
    m->ssd_pf_.assign(hp.n_layer, SsdPrefetch{});
    auto host_f32 = [](ggml_tensor* t) {                            // a small F32 tensor's numbers
        std::vector<float> v((size_t)ggml_nelements(t));
        if (t->type != GGML_TYPE_F32) throw std::runtime_error(std::string(ggml_get_name(t)) + " should be F32");
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
        return v;
    };
    for (int i = 0; i < hp.n_layer; ++i) {
        Layer L = bind_layer(i); bind_hot(L, i);
        if (L.down_x && ssd_set.count(L.down_x)) {
            SsdPrefetch& pf = m->ssd_pf_[i];
            pf.parts[0] = L.gate_up_x; pf.parts[1] = L.gate_x; pf.parts[2] = L.up_x; pf.parts[3] = L.down_x;
            pf.n_expert = L.down_x->ne[2];
            L.ssd = &pf;
        }
        if (L.idx_q) {                                              // QSA: its indexer's norms, for the mask-making op
            QsaLayer Q;
            Q.layer = i; Q.ratio = hp.qsa_ratio[i];
            Q.qn = host_f32(L.idx_q_norm); Q.kn = host_f32(L.idx_k_norm);
            Q.keys.assign(std::max(1, opt.parallel), {});
            Q.blocks.assign(std::max(1, opt.parallel), {});
            L.qsa = (int)m->qsa_.size();
            m->qsa_.push_back(std::move(Q));
        }
        m->layers_.push_back(L);
        if (L.qsa >= 0) m->qsa_calls_.push_back({m.get(), L.qsa});
    }
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
        if (hp.hc) {
            // Flash-Next: the draft block ends in its own stream mixer (the model's last mixer stands in for the final
            // norm, so the block has one too). Its attention runs without the QSA indexer: the same as with it within
            // the token budget; past it, the draft block attends to everything (a guess, checked anyway).
            L.hc_head_norm = T(b + "nextn.hc_head_norm.weight");
            L.hc_head_down = T(b + "nextn.hc_head_down.weight");
            L.hc_head_up   = T(b + "nextn.hc_head_up.weight");
            L.idx_q = L.idx_k = L.idx_q_norm = L.idx_k_norm = nullptr;
        }
        m->mtp_ = std::make_unique<Layer>(L);
    }

    // ---- KV cache on the same device, in f16: half the size and memory traffic of f32 ----
    const int n_seq = std::max(1, opt.parallel);
    m->kv_bufts_.assign(bufts.begin(), bufts.end());
    m->kv_bufts_.push_back(cpu_buft);
    m->make_kv(m->max_ctx_, n_seq);
    for (int i = 0; i < hp.n_layer; ++i) if (m->layer_dev_[i] >= 0) m->place_.gpus[m->layer_dev_[i]].layers++;
    m->cell_seq_.assign(m->max_ctx_, -1);
    m->cell_pos_.assign(m->max_ctx_, -1);
    m->seq_cells_.assign(n_seq, {});
    m->seq_rope_.assign(n_seq, {});
    m->seq_tok_.assign(n_seq, {});
    m->seq_next_.assign(n_seq, 0);
    m->n_free_ = m->max_ctx_;
    // Several conversations in one step: a DeltaNet layer runs once per conversation, so the graph has more nodes.
    if (hp.hybrid && n_seq > 1) m->graph_size_ = 16384 + n_seq * hp.n_layer * 48;

    // Fused flash attention reads the cache once, with no copies. TIERLM_NO_FA=1 uses the plain version.
    m->flash_attn_ = opt.flash_attn >= 0 ? opt.flash_attn == 1 : std::getenv("TIERLM_NO_FA") == nullptr;
    if (ggml_is_quantized((ggml_type)m->kv_t_)) m->flash_attn_ = true;               // only flash attention reads a quantized cache

    // ---- scheduler: runs each op on the device, falling back to the CPU where needed ----
    std::vector<ggml_backend_t> backends(m->gpus_.begin(), m->gpus_.end());
    backends.push_back(m->cpu_);
    const int n_backends = (int)backends.size();
    // The weights in system RAM stay there (see keep_in_ram): not the SSD tier's or other file-mapped parts.
    {
        std::vector<std::pair<const void*, size_t>> ranges;
        for (ggml_backend_buffer_t b : m->wbufs_) {
            if (!b) continue;
            ggml_backend_dev_t d = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(b));
            if (!d || ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_CPU) continue;
            const char* base = (const char*)ggml_backend_buffer_get_base(b);
            const size_t n = ggml_backend_buffer_get_size(b);
            bool mapped = false;
            for (const auto& mf : m->maps_) if (base >= mf->data() && base < mf->data() + mf->size()) mapped = true;
            if (base && n && !mapped) ranges.push_back({base, n});
        }
        m->kept_ram_ = keep_in_ram(ranges);
    }
    // With weights split between VRAM and RAM, let the scheduler copy RAM weights to the GPU for big
    // batches (reading a long prompt), where that beats computing them on the CPU.
    const bool offload = m->main_ != m->cpu_ && (m->place_.cpu_weights > 0 || !m->place_.kv_on_gpu);
    // Layer split: pin each layer's ops to its device (RAM layers to the CPU; with several GPUs, each layer to its
    // GPU too, so ops without weights don't wander to another card).
    m->pin_layers_ = m->main_ != m->cpu_ && m->place_.mode == "layers" && (m->place_.kv_gpu_layers < hp.n_layer || G > 1);
    // The CPU's working memory in pinned (page-locked) host memory from the first GPU's engine, when that's CUDA:
    // the GPU-to-CPU and CPU-to-GPU copies of each split layer's numbers (two per layer per word) then skip
    // the driver's staging copy. TIERLM_PINNED=0 uses plain memory, =1 pinned with any GPU engine that offers it.
    std::vector<ggml_backend_buffer_type_t> sched_bufts;
    for (ggml_backend_t b : backends) sched_bufts.push_back(ggml_backend_get_default_buffer_type(b));
    {
        const char* e = std::getenv("TIERLM_PINNED");
        ggml_backend_dev_t gd = G > 0 ? ggml_backend_get_device(m->gpus_[0]) : nullptr;
        const bool cuda = gd && std::strncmp(ggml_backend_dev_name(gd), "CUDA", 4) == 0;
        const bool want = e ? std::strcmp(e, "0") != 0 : cuda;
        ggml_backend_buffer_type_t host = gd && want ? ggml_backend_dev_host_buffer_type(gd) : nullptr;
        if (host && ggml_backend_supports_buft(m->cpu_, host)) sched_bufts.back() = host;
    }
    m->sched_ = ggml_backend_sched_new(backends.data(), sched_bufts.data(), n_backends, m->graph_size_, false, offload);
    if (!m->sched_) throw std::runtime_error("failed to create the scheduler");
    // CPU and GPU at once on split expert layers (TIERLM_OVERLAP=0: one after the other, as before)
    // (TIERLM_OVERLAP=2, for tests: the same graph, but each step still waits for the one before)
    { const char* e = std::getenv("TIERLM_OVERLAP"); m->overlap_ = !(e && !std::strcmp(e, "0")); }
    ggml_backend_sched_set_overlap(m->sched_, m->overlap_ && !(std::getenv("TIERLM_OVERLAP") && !std::strcmp(std::getenv("TIERLM_OVERLAP"), "2")));
    m->graph_mem_size_ = ggml_tensor_overhead() * m->graph_size_ + ggml_graph_overhead_custom(m->graph_size_, false);

    // Reserve compute memory once, for the largest graph we will ever build (a full batch at the end
    // of the context), so it is not re-planned on every token.
    const int n_res = std::min(m->batch_, m->max_ctx_);
    Graph worst = m->build(n_res, m->max_ctx_, true);
    m->pin_cpu_layers(worst);
    bool ok = ggml_backend_sched_reserve(m->sched_, worst.gf);
    ggml_free(worst.ctx);
    if (!ok) throw std::runtime_error("failed to reserve compute memory");
    if (!m->gpus_.empty()) m->place_.compute_gpu = ggml_backend_sched_get_buffer_size(m->sched_, m->gpus_[0]);
    if (std::getenv("TIERLM_PROFILE") && !m->gpus_.empty())
        std::fprintf(stderr, "profile: working memory for the largest batch: %.0f MB of VRAM\n", m->place_.compute_gpu / 1048576.0);
    // A draft block: schedulers of their own for the check and for the block's one-token step (see Plan), with
    // working memory for a check of up to 8 guesses. Counted with the compute memory, so the VRAM kept free covers it.
    if (m->mtp_ && m->flash_attn_ && !(std::getenv("TIERLM_PLANS") && !std::strcmp(std::getenv("TIERLM_PLANS"), "0")) &&
        !std::getenv("TIERLM_DUMP_SCHED")) {
        auto make = [&](Plan& p, bool mtp) {
            p.sched = ggml_backend_sched_new(backends.data(), sched_bufts.data(), n_backends, m->graph_size_, false, offload);
            if (!p.sched) return;
            ggml_backend_sched_set_overlap(p.sched, m->overlap_ && !(std::getenv("TIERLM_OVERLAP") && !std::strcmp(std::getenv("TIERLM_OVERLAP"), "2")));
            p.mem.resize(m->graph_mem_size_);
            // (4 guesses and the token before them: the default, and the most Find the fastest settings tries; a
            // wider check makes its scheduler grow, as the one scheduler did before. TIERLM_PLAN_WIDTH: another width.)
            const char* we = std::getenv("TIERLM_PLAN_WIDTH");
            const int w = std::min(we ? std::max(2, std::atoi(we)) : 5, m->max_ctx_);
            Graph g = mtp ? m->build_mtp(1, m->max_ctx_ - 1, &p.mem) : m->build(w, m->max_ctx_, true, w, true, w, nullptr, &p.mem);
            if (!mtp) m->pin_cpu_layers(g, p.sched);
            const bool fits = ggml_backend_sched_reserve(p.sched, g.gf);
            ggml_free(g.ctx);
            // Not when it would take more than 1/32 of the card (a check keeps a copy of every DeltaNet layer's memory
            // per guess: 571 MB for Flash-Next, 1.8% of an RTX 5090, for 11% more speed there; on a small card the
            // same memory holds weights that matter more). TIERLM_PLANS=1: always.
            size_t vfree = 0, vtotal = 0;
            m->device_memory(vfree, vtotal);
            const bool big = fits && !m->gpus_.empty() && vtotal > 0 && !std::getenv("TIERLM_PLANS") &&
                             ggml_backend_sched_get_buffer_size(p.sched, m->gpus_[0]) > vtotal / 32;
            if (!fits || big) { ggml_backend_sched_free(p.sched); p.sched = nullptr; p.mem = {}; return; }   // (as before)
            if (!m->gpus_.empty()) m->place_.compute_gpu += ggml_backend_sched_get_buffer_size(p.sched, m->gpus_[0]);
            if (std::getenv("TIERLM_PROFILE") && !m->gpus_.empty())
                std::fprintf(stderr, "profile: the %s's own working memory: %.0f MB of VRAM\n", mtp ? "draft block step" : "check",
                             ggml_backend_sched_get_buffer_size(p.sched, m->gpus_[0]) / 1048576.0);
        };
        make(m->check_, false);
        make(m->mtp1_, true);
    }
    return m;
}

// ggml's scheduler gives any op without weights of its own (norms, RoPE, attention over the cache...) to the
// GPU if it can run it, even in the middle of a layer whose weights are in RAM. Each such op costs two
// GPU<->CPU hand-offs, which on Windows cost more than the GPU saves. So in a layer split, every op of a RAM
// layer is pinned to the CPU, and the work changes device once per token.
void Model::pin_cpu_layers(const Graph& g, ggml_backend_sched* sc) {
    if (!sc) sc = sched_;
    // Debugging: TIERLM_CPU_OPS=SSM_CONV,GATED_DELTA_NET,... runs those op types on the CPU, to find which
    // GPU kernel disagrees with the reference.
    if (const char* ops = std::getenv("TIERLM_CPU_OPS"); ops && main_ != cpu_) {
        const std::string list = std::string(",") + ops + ",";
        for (int i = 0; i < ggml_graph_n_nodes(g.gf); ++i) {
            ggml_tensor* t = ggml_graph_node(g.gf, i);
            if (list.find(std::string(",") + ggml_op_name(t->op) + ",") != std::string::npos)
                ggml_backend_sched_set_tensor_backend(sc, t, cpu_);
        }
    }
    for (ggml_tensor* t : g.cpu_ops) ggml_backend_sched_set_tensor_backend(sc, t, cpu_);
    if (!pin_layers_) return;
    for (const auto& [t, layer] : g.owner) {
        if (layer < 0 || layer >= (int)layer_dev_.size()) continue;
        const int d = layer_dev_[layer];
        if (d < 0) ggml_backend_sched_set_tensor_backend(sc, t, cpu_);
        else if (gpus_.size() > 1) ggml_backend_sched_set_tensor_backend(sc, t, gpus_[d]);
    }
}

// ---- the cache's cells, shared by the conversations ----
std::vector<int> Model::alloc_cells(int s, int n) {
    if (n > n_free_) throw std::runtime_error("context full (" + std::to_string(max_ctx_) + " tokens" +
                                              (n_seq_max() > 1 ? ", shared by all conversations)" : ")"));
    std::vector<int> got;
    got.reserve(n);
    auto& mine = seq_cells_[s];
    for (int c = free_lo_; (int)got.size() < n; ++c) {
        if (c >= (int)cell_seq_.size()) throw std::runtime_error("context cells out of step (no free cell found)");
        if (cell_seq_[c] >= 0) continue;
        cell_seq_[c] = s;
        cell_pos_[c] = (int)mine.size();
        mine.push_back(c);
        got.push_back(c);
        free_lo_ = c + 1;
        hi_ = std::max(hi_, c + 1);
    }
    n_free_ -= n;
    return got;
}

void Model::free_cells_from(int s, int n) {
    auto& mine = seq_cells_[s];
    n = std::max(0, n);
    for (int i = n; i < (int)mine.size(); ++i) {
        const int c = mine[i];
        cell_seq_[c] = cell_pos_[c] = -1;
        free_lo_ = std::min(free_lo_, c);
        ++n_free_;
    }
    if ((int)mine.size() > n) mine.resize(n);
    while (hi_ > 0 && cell_seq_[hi_ - 1] < 0) --hi_;
    rope_truncate(s, n);
}

// Images: the 3-part positions of a conversation's first n positions are kept; the next text position is one
// past the largest of them.
void Model::rope_truncate(int s, int n) {
    if (s >= (int)seq_rope_.size()) return;
    auto& r = seq_rope_[s];
    if ((int)r.size() > n) r.resize(std::max(0, n));
    if (s < (int)seq_tok_.size() && (int)seq_tok_[s].size() > n) seq_tok_[s].resize(std::max(0, n));
    for (QsaLayer& Q : qsa_)
    {
        if (s < (int)Q.keys.size() && Q.keys[s].size() > (size_t)std::max(0, n) * hp_.qsa_dim) Q.keys[s].resize((size_t)std::max(0, n) * hp_.qsa_dim);
        const size_t nb = (size_t)std::max(0, n) / Q.ratio;     // blocks still whole
        if (s < (int)Q.blocks.size() && Q.blocks[s].size() > nb * hp_.qsa_dim) Q.blocks[s].resize(nb * hp_.qsa_dim);
    }
    int next = 0;
    for (const auto& p : r) next = std::max(next, std::max(p[0], std::max(p[1], p[2])) + 1);
    seq_next_[s] = next;
}

const Model::ImageRec* Model::image_of(int32_t id) const {
    for (const ImageRec& im : images_)
        if (id >= im.first && id < im.first + im.rows * im.cols) return &im;
    return nullptr;
}

int32_t Model::find_image(uint64_t key) {
    std::lock_guard<std::recursive_mutex> dev_lock(device_mutex());   // steps read the images under this lock
    for (ImageRec& im : images_) if (im.key == key) { im.used = ++image_clock_; return im.first; }
    return -1;
}

int32_t Model::add_image(uint64_t key, std::vector<float> words, int rows, int cols) {
    if (rows <= 0 || cols <= 0 || words.size() != (size_t)rows * cols * hp_.n_embd)
        throw std::runtime_error("image words don't match the model (is the vision file for this model?)");
    std::lock_guard<std::recursive_mutex> dev_lock(device_mutex());
    if (const int32_t f = find_image(key); f >= 0) return f;
    // Forget the least recently used images beyond ~1 GB (they can be added again; cached conversations that
    // used them simply read the new ids again).
    size_t bytes = words.size() * sizeof(float);
    for (const ImageRec& im : images_) bytes += im.words.size() * sizeof(float);
    while (bytes > ((size_t)1 << 30) && !images_.empty()) {
        auto old = std::min_element(images_.begin(), images_.end(), [](const ImageRec& a, const ImageRec& b) { return a.used < b.used; });
        bytes -= old->words.size() * sizeof(float);
        images_.erase(old);
    }
    if ((int64_t)next_image_id_ + (int64_t)rows * cols >= INT32_MAX) next_image_id_ = kImageIdBase;   // (never in practice)
    ImageRec im;
    im.key = key; im.first = next_image_id_; im.rows = rows; im.cols = cols; im.words = std::move(words); im.used = ++image_clock_;
    next_image_id_ += rows * cols;
    images_.push_back(std::move(im));
    return images_.back().first;
}

int Model::cells_end() const { return hi_; }

void Model::clear_state(int s) {
    std::vector<uint8_t> zeros;
    for (Layer& L : layers_)
        for (ggml_tensor* t : {L.recurrent ? L.conv_states[s] : nullptr, L.recurrent ? L.ssm_states[s] : nullptr,
                               s < (int)L.ple_states.size() ? L.ple_states[s] : nullptr})
            if (t) {
                zeros.assign(ggml_nbytes(t), 0);
                ggml_backend_tensor_set(t, zeros.data(), 0, zeros.size());
            }
}

// Each token sees the cells of its own conversation up to its own position (sliding-window layers: only the last
// n_swa of those). Cells of other conversations, free cells and the padding past the last used cell: -inf.
void Model::fill_masks(Graph& g, int n_kv, const std::vector<int>& tok_seq, const std::vector<int>& tok_pos) {
    const int n_tok = (int)tok_seq.size();
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> mask((size_t)n_kv * n_tok, ninf);
    const int end = std::min(n_kv, hi_);
    for (int t = 0; t < n_tok; ++t) {
        ggml_fp16_t* row = mask.data() + (size_t)t * n_kv;
        const int s = tok_seq[t], p = tok_pos[t];
        // Usually a conversation's cells: walk its own list (positions 0..p), not the whole cache.
        const auto& mine = seq_cells_[s];
        for (int q = 0; q <= p && q < (int)mine.size(); ++q) if (mine[q] < end) row[mine[q]] = zero;
    }
    if (g.mask) ggml_backend_tensor_set(g.mask, mask.data(), 0, sizeof(ggml_fp16_t) * mask.size());
    if (g.mask_swa) {
        for (int t = 0; t < n_tok; ++t) {
            ggml_fp16_t* row = mask.data() + (size_t)t * n_kv;
            const auto& mine = seq_cells_[tok_seq[t]];
            for (int q = 0; q <= tok_pos[t] - hp_.n_swa && q < (int)mine.size(); ++q) if (mine[q] < end) row[mine[q]] = ninf;
        }
        ggml_backend_tensor_set(g.mask_swa, mask.data(), 0, sizeof(ggml_fp16_t) * mask.size());
    }
}

void Model::seq_reset(int s) {
    if (s == 0) { reset(); return; }
    std::lock_guard<std::recursive_mutex> dev_lock(device_mutex());
    free_cells_from(s, 0);
    if (hp_.hybrid) clear_state(s);
}

void Model::seq_truncate(int s, int n) {
    if (s == 0) { truncate(n); return; }
    if (n >= seq_past(s)) return;
    if (hp_.hybrid) { seq_reset(s); return; }                  // the running memory can't be rewound
    std::lock_guard<std::recursive_mutex> dev_lock(device_mutex());   // (another conversation's step uses the cell tables)
    free_cells_from(s, n);
}

void Model::reset() {
    release_pending();
    std::lock_guard<std::recursive_mutex> dev_lock(device_mutex());
    n_past_ = 0;
    free_cells_from(0, 0);
    if (hp_.hybrid) clear_state(0);                             // clear the running memory
    ckpts_.clear();
}

void Model::checkpoint() {
    if (!hp_.hybrid) return;
    release_pending();
    std::lock_guard<std::recursive_mutex> dev_lock(device_mutex());
    while (!ckpts_.empty() && ckpts_.back().n_past >= n_past_) ckpts_.pop_back();   // (one here already, or past the end: replaced)
    Snapshot ck;
    ck.n_past = n_past_;
    auto grab = [&](ggml_tensor* t) {
        ck.parts.emplace_back(ggml_nbytes(t));
        ggml_backend_tensor_get(t, ck.parts.back().data(), 0, ggml_nbytes(t));
        ck.bytes += ggml_nbytes(t);
    };
    for (const auto& L : layers_) {
        if (L.conv_state) { grab(L.conv_state); grab(L.ssm_state_t); }
        if (!L.ple_states.empty()) grab(L.ple_states[0]);
    }
    if (mtp_ && mtp_->conv_state) { grab(mtp_->conv_state); grab(mtp_->ssm_state_t); }
    ckpts_.push_back(std::move(ck));
    // Too many, or too much RAM: the one whose neighbours are closest together goes (never the first or the last),
    // so the ones kept stay spread over the conversation.
    static const size_t max_n = [] { const char* e = std::getenv("TIERLM_ANCHORS"); return (size_t)std::max(1, e ? std::atoi(e) : 12); }();
    static const size_t max_bytes = [] { const char* e = std::getenv("TIERLM_ANCHORS_MB"); return (size_t)(e ? std::atoll(e) : 1536) << 20; }();
    auto total = [&] { size_t n = 0; for (const Snapshot& c : ckpts_) n += c.bytes; return n; };
    while (ckpts_.size() > 1 && (ckpts_.size() > max_n || total() > max_bytes)) {
        size_t drop = 0;
        int gap = INT_MAX;
        for (size_t i = 1; i + 1 < ckpts_.size(); ++i) {
            const int g = ckpts_[i + 1].n_past - ckpts_[i - 1].n_past;
            if (g < gap) { gap = g; drop = i; }
        }
        ckpts_.erase(ckpts_.begin() + (std::ptrdiff_t)drop);       // (two left: the earlier one)
    }
}

void Model::truncate(int n) {
    release_pending();
    if (n >= n_past_) return;
    if (hp_.hybrid) {                                           // the running memory can't be rewound...
        while (!ckpts_.empty() && (ckpts_.back().n_past > n || ckpts_.back().n_past > n_past_)) ckpts_.pop_back();
        if (ckpts_.empty()) { reset(); return; }
        const Snapshot& ck = ckpts_.back();                     // ...but it can go back to the last checkpoint before n
        std::lock_guard<std::recursive_mutex> dev_lock(device_mutex());
        n_past_ = ck.n_past;
        free_cells_from(0, n_past_);
        rope_truncate(0, n_past_);
        size_t i = 0;
        auto put = [&](ggml_tensor* t) { const auto& p = ck.parts.at(i++); ggml_backend_tensor_set(t, p.data(), 0, p.size()); };
        for (const auto& L : layers_) {
            if (L.conv_state) { put(L.conv_state); put(L.ssm_state_t); }
            if (!L.ple_states.empty()) put(L.ple_states[0]);
        }
        if (mtp_ && mtp_->conv_state) { put(mtp_->conv_state); put(mtp_->ssm_state_t); }
        return;
    }
    std::lock_guard<std::recursive_mutex> dev_lock(device_mutex());   // (another conversation's step uses the cell tables)
    n_past_ = n < 0 ? 0 : n;
    free_cells_from(0, n_past_);
}

// Conversation 0's positions, in order, as runs of consecutive cache cells: {first cell, count} (one run when it
// has the cache to itself). The KV cache keeps cell c at offset c * nb[2] in each K and V tensor; the MTP block's
// cache is by position (one run from 0). DeltaNet layers' memories are small and copied whole.
static std::vector<std::pair<int, int>> runs_of(const std::vector<int>& cells) {
    std::vector<std::pair<int, int>> r;
    for (int c : cells) {
        if (!r.empty() && r.back().first + r.back().second == c) ++r.back().second;
        else r.push_back({c, 1});
    }
    return r;
}

Model::Snapshot Model::save() const {
    std::lock_guard<std::recursive_mutex> dev_lock(device_mutex());
    Snapshot s;
    s.n_past = n_past_;
    s.rope = seq_rope_[0];
    if (!seq_tok_.empty()) s.tok = seq_tok_[0];
    for (const QsaLayer& Q : qsa_) s.qsa.push_back(Q.keys[0]);
    const auto runs = runs_of(seq_cells_[0]);
    const std::vector<std::pair<int, int>> by_pos(n_past_ ? 1 : 0, {0, n_past_});
    auto grab = [&](ggml_tensor* t, size_t n) {
        s.parts.emplace_back(n);
        if (n) ggml_backend_tensor_get(t, s.parts.back().data(), 0, n);
        s.bytes += n;
    };
    auto grab_rows = [&](ggml_tensor* t, const std::vector<std::pair<int, int>>& rs) {
        const size_t row = t->nb[2];
        s.parts.emplace_back((size_t)n_past_ * row);
        size_t off = 0;
        for (auto [c, n] : rs) { ggml_backend_tensor_get(t, s.parts.back().data() + off, (size_t)c * row, (size_t)n * row); off += (size_t)n * row; }
        s.bytes += s.parts.back().size();
    };
    auto layer = [&](const Layer& L, bool mtp) {
        if (L.conv_state) { grab(L.conv_state, ggml_nbytes(L.conv_state)); grab(L.ssm_state_t, ggml_nbytes(L.ssm_state_t)); }
        if (!L.ple_states.empty()) grab(L.ple_states[0], ggml_nbytes(L.ple_states[0]));
        if (L.k_cache) { grab_rows(L.k_cache, mtp ? by_pos : runs); if (L.v_cache) grab_rows(L.v_cache, mtp ? by_pos : runs); }
    };
    for (const auto& L : layers_) layer(L, false);
    if (mtp_) layer(*mtp_, true);
    return s;
}

void Model::restore(const Snapshot& s) {
    release_pending();
    ckpts_.clear();
    std::lock_guard<std::recursive_mutex> dev_lock(device_mutex());
    n_past_ = 0;
    free_cells_from(0, 0);
    alloc_cells(0, s.n_past);                                     // throws if the other conversations left no room
    n_past_ = s.n_past;
    seq_rope_[0] = s.rope;
    if ((int)seq_rope_[0].size() != s.n_past) {                   // (saved by a model without 3-part positions)
        seq_rope_[0].resize(s.n_past);
        for (int i = 0; i < s.n_past; ++i) seq_rope_[0][i] = {i, i, i};
    }
    rope_truncate(0, s.n_past);
    if (!seq_tok_.empty()) {
        if (hp_.any_ple() && (int)s.tok.size() != s.n_past) throw std::runtime_error("saved conversation doesn't match this model");
        seq_tok_[0] = s.tok;
    }
    if (!qsa_.empty()) {
        if (s.qsa.size() != qsa_.size()) throw std::runtime_error("saved conversation doesn't match this model");
        for (size_t i = 0; i < qsa_.size(); ++i) { qsa_[i].keys[0] = s.qsa[i]; qsa_[i].blocks[0].clear(); }   // (made again after the next step)
    }
    const auto runs = runs_of(seq_cells_[0]);
    const std::vector<std::pair<int, int>> by_pos(s.n_past ? 1 : 0, {0, s.n_past});
    size_t i = 0;
    auto next = [&](ggml_tensor* t) -> const std::vector<uint8_t>& {
        if (i >= s.parts.size()) throw std::runtime_error("saved conversation doesn't match this model");
        const auto& p = s.parts[i++];
        if (p.size() > ggml_nbytes(t)) throw std::runtime_error("saved conversation doesn't match this model");
        return p;
    };
    auto put = [&](ggml_tensor* t) { const auto& p = next(t); if (!p.empty()) ggml_backend_tensor_set(t, p.data(), 0, p.size()); };
    auto put_rows = [&](ggml_tensor* t, const std::vector<std::pair<int, int>>& rs) {
        const auto& p = next(t);
        const size_t row = t->nb[2];
        if (p.size() != (size_t)s.n_past * row) throw std::runtime_error("saved conversation doesn't match this model");
        size_t off = 0;
        for (auto [c, n] : rs) { ggml_backend_tensor_set(t, p.data() + off, (size_t)c * row, (size_t)n * row); off += (size_t)n * row; }
    };
    auto layer = [&](const Layer& L, bool mtp) {
        if (L.conv_state) { put(L.conv_state); put(L.ssm_state_t); }
        if (!L.ple_states.empty()) put(L.ple_states[0]);
        if (L.k_cache) { put_rows(L.k_cache, mtp ? by_pos : runs); if (L.v_cache) put_rows(L.v_cache, mtp ? by_pos : runs); }
    };
    for (const auto& L : layers_) layer(L, false);
    if (mtp_) layer(*mtp_, true);
    if (i != s.parts.size()) throw std::runtime_error("saved conversation doesn't match this model");
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
// Multi-head latent attention (DeepSeek V2/V3, Kimi K2.x): as Hugging Face's DeepseekV3Attention computes it, but on
// the compressed cache. Hugging Face expands every cached token into full per-head keys and values (kv_b); here the
// same products are taken in the compressed space instead: each head's query (its unrotated part) goes through k_b
// into the kv_lora-sized space, attention runs over the cached [compressed | rotated] vectors (one "head" shared by
// all, as in multi-query attention), and each head's result goes through v_b. q.k and the weighted sum of v are the
// same sums in another order, so the result is the same; the cache is kv_lora + n_rot numbers per token per layer.
ggml_tensor* Model::attn_mla(ggml_context* c, Graph& g, const Layer& L, ggml_tensor* h, int n_tok, int n_past) {
    const Hparams& hp = hp_;
    const int H = hp.n_head, R = hp.kv_lora, P = hp.n_rot, N = hp.nope, Vd = hp.v_dim, n_kv = kv_span(n_past + n_tok);
    // queries: [N + P, H, n_tok]
    ggml_tensor* q = L.wq ? ggml_mul_mat(c, L.wq, h)
                          : ggml_mul_mat(c, L.wq_b, ggml_mul(c, ggml_rms_norm(c, ggml_mul_mat(c, L.wq_a, h), hp.rms_eps), L.q_a_norm));
    q = ggml_reshape_3d(c, q, N + P, H, n_tok);
    const size_t es = ggml_element_size(q);
    ggml_tensor* q_nope = ggml_view_3d(c, q, N, H, n_tok, q->nb[1], q->nb[2], 0);
    ggml_tensor* q_pe   = ggml_view_3d(c, q, P, H, n_tok, q->nb[1], q->nb[2], (size_t)N * es);
    // this token's compressed key/value (normalised) and shared rotated key: [R + P, n_tok]
    ggml_tensor* kv = ggml_mul_mat(c, L.wkv_a, h);
    ggml_tensor* ckv = ggml_view_2d(c, kv, R, n_tok, kv->nb[1], 0);
    ckv = ggml_mul(c, ggml_rms_norm(c, ggml_cont(c, ckv), hp.rms_eps), L.kv_a_norm);
    ggml_tensor* k_pe = ggml_view_3d(c, kv, P, 1, n_tok, kv->nb[1], kv->nb[1], (size_t)R * ggml_element_size(kv));
    ggml_tensor* ff = g.rope_ff ? g.rope_ff : rope_freqs_;
    q_pe = ggml_rope_ext(c, ggml_cont(c, q_pe), g.pos, ff, P, 0, hp.n_ctx, hp.rope_base, hp.rope_scale, 0.0f, hp.rope_attn_factor, 32.0f, 1.0f);
    k_pe = ggml_rope_ext(c, ggml_cont(c, k_pe), g.pos, ff, P, 0, hp.n_ctx, hp.rope_base, hp.rope_scale, 0.0f, hp.rope_attn_factor, 32.0f, 1.0f);
    // into the cache: [R + P] per token
    ggml_tensor* krow = ggml_concat(c, ckv, ggml_reshape_2d(c, k_pe, P, n_tok), 0);
    ggml_tensor* krows = ggml_reshape_2d(c, L.k_cache, R + P, L.k_cache->ne[2]);
    ggml_tensor* where = g.cells ? g.cells : g.pos;
    ggml_build_forward_expand(g.gf, ggml_set_rows(c, krows, krow, where));
    // each head's query in the compressed space: k_b[h] (N x R) applied to its unrotated part -> [R, n_tok, H]
    ggml_tensor* qn = ggml_permute(c, q_nope, 0, 2, 1, 3);                // [N, n_tok, H]
    ggml_tensor* kb = L.wk_b->ne[2] == H ? L.wk_b : ggml_reshape_3d(c, L.wk_b, N, R, H);
    ggml_tensor* q_lat = ggml_mul_mat(c, kb, qn);                         // [R, n_tok, H]
    ggml_tensor* Q = ggml_concat(c, q_lat, ggml_permute(c, q_pe, 0, 2, 1, 3), 0);   // [R + P, n_tok, H]
    ggml_tensor* kall = ggml_view_3d(c, L.k_cache, R + P, 1, n_kv, L.k_cache->nb[1], L.k_cache->nb[2], 0);
    ggml_tensor* K = ggml_permute(c, kall, 0, 2, 1, 3);                   // [R + P, n_kv, 1]
    ggml_tensor* vall = ggml_view_3d(c, L.k_cache, R, 1, n_kv, L.k_cache->nb[1], L.k_cache->nb[2], 0);   // the compressed part
    const float scale = hp.kq_scale;
    ggml_tensor* o;                                                       // [R, n_tok, H]
    if (flash_attn_) {
        o = ggml_flash_attn_ext(c, Q, K, ggml_permute(c, vall, 0, 2, 1, 3), g.mask, scale, 0.0f, 0.0f);   // [R, H, n_tok]
        o = ggml_permute(c, o, 0, 2, 1, 3);
    } else {
        ggml_tensor* kq = ggml_mul_mat(c, K, Q);                          // [n_kv, n_tok, H]
        kq = ggml_soft_max_ext(c, kq, g.mask, scale, 0.0f);
        ggml_tensor* Vt = ggml_cont(c, ggml_permute(c, vall, 1, 2, 0, 3)); // [n_kv, R, 1]
        o = ggml_mul_mat(c, Vt, kq);                                      // [R, n_tok, H]
    }
    // each head's result out of the compressed space: v_b[h] (R -> Vd), then all heads through the output projection
    ggml_tensor* vb = L.wv_b->ne[2] == H ? L.wv_b : ggml_reshape_3d(c, L.wv_b, R, Vd, H);
    ggml_tensor* ov = ggml_mul_mat(c, vb, ggml_cont(c, o));              // [Vd, n_tok, H]
    ov = ggml_cont(c, ggml_permute(c, ov, 0, 2, 1, 3));                   // [Vd, H, n_tok]
    return ggml_mul_mat(c, L.wo, ggml_reshape_2d(c, ov, (int64_t)Vd * H, n_tok));
}

// ---- QSA: Flash-Next's sparse attention (Hugging Face's Qwen4ExpTextQSAIndexer) ----
// For each query: its conversation's visible positions are cut into complete blocks of `ratio` tokens; each block's
// key is the mean of its tokens' indexer keys, normed (k_norm) and rotated to the block's first position. The query's
// indexer heads, normed (q_norm) and rotated, score each block: sum over heads of relu(q . key) / sqrt(dim). The
// budget / ratio best blocks are kept, with the positions after the last complete block; everything else this query
// could see is masked out. With no more complete blocks than that, all are kept: plain attention.
static void qsa_op(ggml_tensor* dst, const ggml_tensor* a, const ggml_tensor* b, const ggml_tensor* c, int ith, int nth, void* ud) {
    const auto* call = (const Model::QsaCall*)ud;
    call->m->qsa_mask(call->m->qsa_layer(call->i), a, b, c, dst, ith, nth);
}

// One block's key: the mean of its tokens' raw indexer keys, normed (k_norm) and rotated to its first position.
static void qsa_rotate(const Hparams& hp, float* v, const std::array<int, 3>& pos) {
    const int P = hp.n_rot, half = P / 2;                    // NeoX halves of the first n_rot numbers, 3-part positions
    for (int i = 0; i < half; ++i) {
        int which = 0;                                       // Hugging Face's recomposition_frequencies (T, then H, W)
        if (hp.mrope) {
            if (i % 3 == 1 && i < 3 * hp.mrope_sections[1]) which = 1;
            else if (i % 3 == 2 && i < 3 * hp.mrope_sections[2]) which = 2;
        }
        const double th = pos[which] * std::pow((double)hp.rope_base, -2.0 * i / P);
        const float cs = (float)std::cos(th), sn = (float)std::sin(th), x0 = v[i], x1 = v[i + half];
        v[i] = x0 * cs - x1 * sn;
        v[i + half] = x1 * cs + x0 * sn;
    }
}
static void qsa_rms(const Hparams& hp, float* v, const std::vector<float>& w) {
    const int D = hp.qsa_dim;
    double ss = 0; for (int i = 0; i < D; ++i) ss += (double)v[i] * v[i];
    const float r = (float)(1.0 / std::sqrt(ss / D + hp.rms_eps));
    for (int i = 0; i < D; ++i) v[i] = v[i] * r * w[i];
}
void Model::qsa_block_key(const QsaLayer& Q, const float* const* raw, const std::array<int, 3>& pos, float* out) const {
    const int D = hp_.qsa_dim;
    for (int i = 0; i < D; ++i) out[i] = 0.0f;
    for (int j = 0; j < Q.ratio; ++j) for (int i = 0; i < D; ++i) out[i] += raw[j][i];
    for (int i = 0; i < D; ++i) out[i] /= Q.ratio;
    qsa_rms(hp_, out, Q.kn);
    qsa_rotate(hp_, out, pos);
}

void Model::qsa_mask(const QsaLayer& Q, const ggml_tensor* mask, const ggml_tensor* qraw, const ggml_tensor* kraw, ggml_tensor* dst,
                     int ith, int nth) const {
    const Hparams& hp = hp_;
    const int n_kv = (int)mask->ne[0], n_tok = (int)mask->ne[1], D = hp.qsa_dim, NH = hp.qsa_heads, R = Q.ratio;
    const int top = hp.qsa_budget / R;
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
    const QsaBatch& B = qsa_batch_;
    std::vector<float> key(D), qh((size_t)NH * D), score;
    std::vector<int> order;
    for (int t = 0; t < n_tok; ++t) {
        // rows split between threads; each copies its rows of the mask first
        if (t % nth != ith) continue;
        std::memcpy((char*)dst->data + (size_t)t * dst->nb[1], (const char*)mask->data + (size_t)t * mask->nb[1], (size_t)n_kv * sizeof(ggml_fp16_t));
        const int s = B.seq[t], p = B.pos[t], vis = p + 1, nb = vis / R;
        if (nb <= top) continue;                             // within the budget: plain attention
        const auto& stored = Q.keys[s];
        auto raw = [&](int q) -> const float* {              // a position's raw indexer key: stored, or from this batch
            if ((size_t)(q + 1) * D <= stored.size() && !B.at.count({s, q})) return stored.data() + (size_t)q * D;
            const int bt = B.at.at({s, q});
            return (const float*)((const char*)kraw->data + (size_t)bt * kraw->nb[1]);
        };
        auto rope_of = [&](int q) -> std::array<int, 3> {
            auto it = B.at.find({s, q});
            if (it != B.at.end()) return B.rope[it->second];
            return seq_rope_[s][q];
        };
        // the query's heads, normed and rotated
        for (int hh = 0; hh < NH; ++hh) {
            float* v = qh.data() + (size_t)hh * D;
            std::memcpy(v, (const char*)qraw->data + (size_t)t * qraw->nb[1] + (size_t)hh * D * sizeof(float), D * sizeof(float));
            qsa_rms(hp, v, Q.qn);
            qsa_rotate(hp, v, B.rope[t]);
        }
        score.assign(nb, 0.0f);
        const std::vector<float>& cached = Q.blocks[s];         // blocks finished before this step
        std::vector<const float*> parts(R);
        for (int bi = 0; bi < nb; ++bi) {
            const float* bk;
            if ((size_t)(bi + 1) * D <= cached.size()) bk = cached.data() + (size_t)bi * D;
            else {                                               // a block that this step finished: made here
                for (int j = 0; j < R; ++j) parts[j] = raw(bi * R + j);
                qsa_block_key(Q, parts.data(), rope_of(bi * R), key.data());
                bk = key.data();
            }
            float sc = 0;
            for (int hh = 0; hh < NH; ++hh) {
                const float* v = qh.data() + (size_t)hh * D;
                float d = 0; for (int i = 0; i < D; ++i) d += v[i] * bk[i];
                sc += std::max(0.0f, d);
            }
            score[bi] = sc / std::sqrt((float)D);
        }
        order.resize(nb);
        for (int bi = 0; bi < nb; ++bi) order[bi] = bi;
        std::partial_sort(order.begin(), order.begin() + top, order.end(), [&](int x, int y) { return score[x] > score[y]; });
        std::vector<char> keep(nb, 0);
        for (int k = 0; k < top; ++k) keep[order[k]] = 1;
        ggml_fp16_t* row = (ggml_fp16_t*)((char*)dst->data + (size_t)t * dst->nb[1]);
        for (int q = 0; q < nb * R; ++q)
            if (!keep[q / R]) row[seq_cells_[s][q]] = ninf;
    }
}

ggml_tensor* Model::attn_block(ggml_context* c, Graph& g, const Layer& L, ggml_tensor* h, int n_tok, int n_past) {
    if (hp_.mla) return attn_mla(c, g, L, h, n_tok, n_past);
    const Hparams& hp = hp_;
    const int hd = hp.head_dim, n_kv = kv_span(n_past + n_tok);     // padded: see kv_span
    const float scale = 1.0f / std::sqrt((float)hd);
        // Qwen3.5's attention layers project each head's query together with an output gate: [q | gate]
        ggml_tensor* qg = hp.gated_attn ? ggml_mul_mat(c, L.wq, h) : nullptr;    // [2*hd*n_head, n_tok]
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
        ggml_tensor* ff = g.rope_ff ? g.rope_ff : rope_freqs_;           // YaRN (computed) or Llama 3.1 factors (file)
        if (L.rope) {                                                        // (Flint's full-attention layers: no positions)
            if (g.mrope) {                                                   // images: 3-part positions (interleaved)
                int sec[4] = { hp.mrope_sections[0], hp.mrope_sections[1], hp.mrope_sections[2], hp.mrope_sections[3] };
                q = ggml_rope_multi(c, q, g.pos, ff, hp.n_rot, sec, GGML_ROPE_TYPE_IMROPE, hp.n_ctx, hp.rope_base, hp.rope_scale, 0.0f, hp.rope_attn_factor, 32.0f, 1.0f);
                k = ggml_rope_multi(c, k, g.pos, ff, hp.n_rot, sec, GGML_ROPE_TYPE_IMROPE, hp.n_ctx, hp.rope_base, hp.rope_scale, 0.0f, hp.rope_attn_factor, 32.0f, 1.0f);
            } else {
                q = ggml_rope_ext(c, q, g.pos, ff, hp.n_rot, mode, hp.n_ctx, hp.rope_base, hp.rope_scale, 0.0f, hp.rope_attn_factor, 32.0f, 1.0f);
                k = ggml_rope_ext(c, k, g.pos, ff, hp.n_rot, mode, hp.n_ctx, hp.rope_base, hp.rope_scale, 0.0f, hp.rope_attn_factor, 32.0f, 1.0f);
            }
        }

        // Write this batch's keys/values into the cache at their cells (g.cells; for one conversation alone, its
        // positions n_past, n_past + 1, ...).
        // Written by index rather than into a view at n_past, so the graph is the same from one token to the
        // next (CUDA graphs can replay it). Added to the graph first, so they run before the attention below.
        const size_t knb1 = L.k_cache->nb[1], knb2 = L.k_cache->nb[2];
        const size_t vnb1 = L.v_cache->nb[1], vnb2 = L.v_cache->nb[2];
        ggml_tensor* krows = ggml_reshape_2d(c, L.k_cache, hd * hp.n_head_kv, L.k_cache->ne[2]);
        ggml_tensor* vrows = ggml_reshape_2d(c, L.v_cache, hd * hp.n_head_kv, L.v_cache->ne[2]);
        ggml_tensor* where = g.cells ? g.cells : g.pos;                      // (the MTP block's cache: by position)
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, krows, ggml_reshape_2d(c, k, hd * hp.n_head_kv, n_tok), where));
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, vrows, ggml_reshape_2d(c, v, hd * hp.n_head_kv, n_tok), where));
        ggml_tensor* kall = ggml_view_3d(c, L.k_cache, hd, hp.n_head_kv, n_kv, knb1, knb2, 0);   // all tokens so far, padded
        ggml_tensor* vall = ggml_view_3d(c, L.v_cache, hd, hp.n_head_kv, n_kv, vnb1, vnb2, 0);

        ggml_tensor* Q = ggml_permute(c, q, 0, 2, 1, 3);                     // [hd, n_tok, n_head]
        ggml_tensor* K = ggml_permute(c, kall, 0, 2, 1, 3);                  // [hd, n_kv, n_head_kv]
        ggml_tensor* o;
        ggml_tensor* mask = L.swa ? g.mask_swa : g.mask;                     // sliding-window layers see the last n_swa tokens
        if (L.idx_k) {
            // QSA (Flash-Next): the indexer's keys for this batch are kept (read after the step); a query further than
            // the budget from its conversation's start attends only to the blocks the indexer ranks highest (the
            // mask is made on the CPU, see Model::qsa_mask). Within the budget it is plain attention.
            ggml_tensor* kr = ggml_mul_mat(c, L.idx_k, h);                   // [qsa_dim, n_tok]
            ggml_set_output(kr);
            ggml_build_forward_expand(g.gf, kr);
            g.qsa_k.push_back({L.qsa, kr});
            if (qsa_long_) {
                ggml_tensor* qr = ggml_mul_mat(c, L.idx_q, h);               // [qsa_heads * qsa_dim, n_tok]
                mask = ggml_map_custom3(c, mask, qr, kr, qsa_op, GGML_N_TASKS_MAX, (void*)&qsa_calls_[L.qsa]);
                g.cpu_ops.push_back(mask);
            }
        }
        if (flash_attn_) {
            ggml_tensor* V = ggml_permute(c, vall, 0, 2, 1, 3);              // [hd, n_kv, n_head_kv], no copy
            o = ggml_flash_attn_ext(c, Q, K, V, mask, scale, 0.0f, 0.0f);    // [hd, n_head, n_tok]
            if (L.sinks) ggml_flash_attn_ext_add_sinks(o, L.sinks);          // gpt-oss: one more score per head, never attended to
        } else {
            ggml_tensor* kq = ggml_mul_mat(c, K, Q);                         // [n_kv, n_tok, n_head]
            if (mask) {
                kq = ggml_soft_max_ext(c, kq, mask, scale, 0.0f);            // mask: causal (and window)
                if (L.sinks) ggml_soft_max_add_sinks(kq, L.sinks);
            } else {
                kq = ggml_scale(c, kq, scale);
                kq = ggml_diag_mask_inf(c, kq, n_past);                      // causal: no peeking ahead
                kq = ggml_soft_max(c, kq);
            }
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
        o = ggml_mul_mat(c, L.wo, o);
        return L.bo ? ggml_add(c, o, L.bo) : o;
}

// Feed-forward for one layer: dense SwiGLU, or mixture of experts (with an optional shared expert).
// Returns the output to be added to the residual stream.
// Makes the expert ids of each token distinct (see the hot-expert split in ffn_block). ids is I32
// [n_used, n_tok]; userdata is the number of experts in the part. A repeated id is replaced by the lowest id
// this token does not use yet (the placement keeps every part at least n_used experts big).
static void distinct_ids(ggml_tensor* dst, const ggml_tensor* a, int ith, int nth, void* userdata) {
    if (ith != 0) return;
    (void)nth;
    const int64_t n_used = a->ne[0], n_tok = a->ne[1];
    for (int64_t t = 0; t < n_tok; ++t) {
        const int32_t* in = (const int32_t*)((const char*)a->data + t * a->nb[1]);
        int32_t* out = (int32_t*)((char*)dst->data + t * dst->nb[1]);
        // userdata: the part's size; negative for a RAM part read from the file, whose "no expert here" is the size
        // itself (out of range: it has no zero expert), moved to an unused expert like a repeat (its weight is 0)
        const int32_t n_part = std::abs((int32_t)(intptr_t)userdata);
        for (int64_t k = 0; k < n_used; ++k) out[k] = in[k];
        for (int64_t k = 0; k < n_used; ++k) {
            bool dup = out[k] < 0 || out[k] >= n_part;
            for (int64_t j = 0; j < k; ++j) if (out[j] == out[k]) { dup = true; break; }
            if (!dup) continue;
            for (int32_t cand = 0; cand < n_part; ++cand) {
                bool used = false;
                for (int64_t j = 0; j < n_used; ++j) if (out[j] == cand) { used = true; break; }
                if (!used) { out[k] = cand; break; }
            }
        }
    }
}

// For the RAM part of a split layer when the CPU computes it (a few tokens): picks that point at the zero expert
// (the part's last one; userdata is the part's size) become -1, which the CPU kernel skips, writing zeros. So
// the CPU computes only the experts that live in RAM, instead of multiplying the zero expert out as well.
static void skip_zero_ids(ggml_tensor* dst, const ggml_tensor* a, int ith, int nth, void* userdata) {
    if (ith != 0) return;
    (void)nth;
    const int32_t u = (int32_t)(intptr_t)userdata;
    const int32_t zero = u > 0 ? u - 1 : -u;                  // negative: read from the file, no zero expert (see distinct_ids)
    for (int64_t t = 0; t < a->ne[1]; ++t) {
        const int32_t* in = (const int32_t*)((const char*)a->data + t * a->nb[1]);
        int32_t* out = (int32_t*)((char*)dst->data + t * dst->nb[1]);
        for (int64_t k = 0; k < a->ne[0]; ++k) out[k] = in[k] == zero ? -1 : in[k];
    }
}

// The SSD tier: once the router has picked a layer's experts, the operating system is asked to start reading each
// chosen expert's weights from the drive (all of them at once, in large reads), before the CPU gets to them, instead
// of the CPU waiting on one small page read after another. The ids pass through unchanged. userdata: the Layer.
static std::atomic<unsigned long long> g_ssd_ahead{0};   // bytes asked for ahead (TIERLM_SSD_LOG=1 prints them)
static void prefetch_experts(ggml_tensor* dst, const ggml_tensor* a, int ith, int nth, void* userdata) {
    (void)nth;
    if (ith != 0) return;
    std::memcpy(dst->data, a->data, ggml_nbytes(a));
    if (std::getenv("TIERLM_SSD_PREFETCH") && std::string(std::getenv("TIERLM_SSD_PREFETCH")) == "0") return;
    const auto* pf = (const SsdPrefetch*)userdata;
    const int64_t n_expert = pf->n_expert;
    std::vector<char> seen((size_t)n_expert, 0);
    for (int64_t t = 0; t < a->ne[1]; ++t) {
        const int32_t* in = (const int32_t*)((const char*)a->data + t * a->nb[1]);
        for (int64_t k = 0; k < a->ne[0]; ++k) {
            const int32_t e = in[k];
            if (e < 0 || e >= n_expert || seen[e]) continue;
            seen[e] = 1;
            for (const ggml_tensor* x : pf->parts)
                if (x) { MappedFile::prefetch((const char*)x->data + (size_t)e * x->nb[2], x->nb[2]); g_ssd_ahead += x->nb[2]; }
        }
    }
    static const bool log = std::getenv("TIERLM_SSD_LOG") != nullptr;
    static auto last = std::chrono::steady_clock::now();
    if (log && (std::chrono::steady_clock::now() - last > std::chrono::seconds(2) || std::string(std::getenv("TIERLM_SSD_LOG")) == "2")) {
        last = std::chrono::steady_clock::now();
        std::fprintf(stderr, "SSD tier: %.2f GB asked for ahead so far\n", g_ssd_ahead.load() / 1e9);
    }
}

ggml_tensor* Model::ffn_block(ggml_context* c, Graph& g, const Layer& L, ggml_tensor* h, int n_tok, bool want_routing) {
    const Hparams& hp = hp_;
        if (!L.router) {
            // ---- dense SwiGLU feed-forward (dense models, and Flint's first layers) ----
            // one SwiGLU step (not silu, then multiply): the GPU then does both multiplies and it in one kernel
            ggml_tensor* act = ggml_swiglu_split(c, ggml_mul_mat(c, L.gate, h), ggml_mul_mat(c, L.up, h));
            return ggml_mul_mat(c, L.down, act);
        }
        // ---- mixture of experts ----
        // (Nemotron: each block has its own number of experts and of experts per token)
        const int NE = hp.n_expert_at(L.index), NU = hp.n_used_at(L.index);
        ggml_tensor* scores = ggml_mul_mat(c, L.router, h);                 // [n_expert, n_tok]
        if (L.router_b) scores = ggml_add(c, scores, L.router_b);
        // Most families: softmax over all experts, then the top ones' probabilities. gpt-oss: the top scores, then
        // softmax over just those (same top experts either way: softmax keeps the order).
        ggml_tensor* probs = hp.expert_sigmoid ? ggml_sigmoid(c, scores)          // Flint: each expert's own sigmoid
                           : hp.expert_softmax_first ? ggml_soft_max(c, scores) : scores;
        // Which experts: by probs, or (DeepSeek V3) by probs plus a per-expert bias, and only from the best groups.
        ggml_tensor* choice = L.exp_probs_b ? ggml_add(c, probs, L.exp_probs_b) : probs;
        if (hp.n_group > 1) {
            // each group's score: the sum of its two best (V3) or its best (V2); groups below the topk_group-th best
            // are shut out by a large negative amount
            const int G = hp.n_group, per = NE / G;
            ggml_tensor* c3 = ggml_reshape_3d(c, ggml_cont(c, choice), per, G, n_tok);
            ggml_tensor* srt = ggml_argsort(c, c3, GGML_SORT_ORDER_DESC);  // [per, G, n_tok]
            const int top = hp.group_sum2 ? std::min(2, per) : 1;
            ggml_tensor* best = ggml_get_rows(c, ggml_reshape_4d(c, c3, 1, per, G, n_tok),
                                              ggml_view_3d(c, srt, top, G, n_tok, srt->nb[1], srt->nb[2], 0));   // [1, top, G, n_tok]
            ggml_tensor* gs = ggml_sum_rows(c, ggml_reshape_3d(c, best, top, G, n_tok));                       // [1, G, n_tok]
            gs = ggml_reshape_2d(c, gs, G, n_tok);
            ggml_tensor* gsrt = ggml_argsort(c, gs, GGML_SORT_ORDER_DESC); // [G, n_tok]
            ggml_tensor* kth = ggml_get_rows(c, ggml_reshape_3d(c, gs, 1, G, n_tok),
                                             ggml_view_2d(c, gsrt, 1, n_tok, gsrt->nb[1], (size_t)(hp.topk_group - 1) * gsrt->nb[0]));   // [1, 1, n_tok]
            ggml_tensor* thr = ggml_reshape_2d(c, kth, 1, n_tok);
            // kept: score >= the topk_group-th best (step of score - threshold, nudged so the threshold group counts)
            ggml_tensor* keep = ggml_step(c, ggml_scale_bias(c, ggml_sub(c, gs, ggml_repeat(c, thr, gs)), 1.0f, 1e-6f));
            ggml_tensor* pen = ggml_scale_bias(c, keep, 1e9f, -1e9f);      // 0 kept, -1e9 shut out
            c3 = ggml_add(c, c3, ggml_repeat(c, ggml_reshape_3d(c, pen, 1, G, n_tok), c3));
            choice = ggml_reshape_2d(c, c3, NE, n_tok);
        }
        ggml_tensor* order = ggml_argsort(c, choice, GGML_SORT_ORDER_DESC); // [n_expert, n_tok] best first
        ggml_tensor* sel   = ggml_view_2d(c, order, NU, n_tok, order->nb[1], 0);   // top n_used
        // Routing for the dashboard and expert learning: just the chosen experts, copied out compactly. The rankings
        // themselves are not made outputs, so the GPU can still fuse the router's steps; all layers are joined and
        // read back in one go at the end (see build), instead of one GPU read per layer.
        if (want_routing) { g.orders.push_back(ggml_cont(c, sel)); g.order_layers.push_back(L.index); }   // [n_used, n_tok]
        ggml_tensor* w     = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, NE, n_tok), sel); // [1, n_used, n_tok]
        if (!hp.expert_softmax_first)
            w = ggml_reshape_3d(c, ggml_soft_max(c, ggml_reshape_2d(c, w, NU, n_tok)), 1, NU, n_tok);
        if (hp.expert_renorm) {                                             // chosen weights sum to 1
            ggml_tensor* w2 = ggml_reshape_2d(c, w, NU, n_tok);
            w2 = ggml_div(c, w2, ggml_sum_rows(c, w2));
            w = ggml_reshape_3d(c, w2, 1, NU, n_tok);
        }
        if (hp.expert_scale != 0.0f && hp.expert_scale != 1.0f) w = ggml_scale(c, w, hp.expert_scale);
        // LatentMoE (Nemotron): the experts read and write a narrower width, between a down and an up projection;
        // the router above and the shared expert below work at the model's.
        ggml_tensor* hx    = L.lat_down ? ggml_mul_mat(c, L.lat_down, h) : h;
        const int64_t XW   = hx->ne[0];
        ggml_tensor* h3    = ggml_reshape_3d(c, hx, XW, 1, n_tok);
        ggml_tensor* xin   = h3;                                            // what the experts read (see overlap below)
        // The chosen experts' outputs, weighted and summed: [n_embd, n_tok]. ids: [n_used, n_tok] into these tensors.
        auto experts = [&](ggml_tensor* gate_up_x, ggml_tensor* gate_x, ggml_tensor* up_x, ggml_tensor* down_x,
                           ggml_tensor* ids, ggml_tensor* wt, ggml_tensor* gate_b = nullptr, ggml_tensor* up_b = nullptr,
                           ggml_tensor* down_b = nullptr, ggml_tensor* bias_ids = nullptr) {
            if (!bias_ids) bias_ids = ids;                                   // (a split layer: the router's own ids, see below)
            ggml_tensor *up, *gate = nullptr;
            if (hp.expert_relu2) up = ggml_mul_mat_id(c, up_x, xin, ids);   // Nemotron: no gate
            else if (gate_up_x) {                                                 // [2*n_ff_exp, n_used, n_tok]: gate, then up
                ggml_tensor* gu = ggml_mul_mat_id(c, gate_up_x, xin, ids);
                const int64_t f = gu->ne[0] / 2;
                gate = ggml_view_3d(c, gu, f, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], 0);
                up   = ggml_view_3d(c, gu, f, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], f * gu->nb[0]);
            } else {
                up   = ggml_mul_mat_id(c, up_x, xin, ids);                    // [n_ff_exp, n_used, n_tok]
                gate = ggml_mul_mat_id(c, gate_x, xin, ids);
            }
            if (up_b)   up   = ggml_add_id(c, up, up_b, bias_ids);            // each pick gets its own expert's bias
            if (gate_b) gate = ggml_add_id(c, gate, gate_b, bias_ids);
            ggml_tensor* act  = hp.expert_relu2 ? ggml_sqr(c, ggml_relu(c, up))
                              : hp.swiglu_oai ? ggml_swiglu_oai(c, gate_up_x ? ggml_cont(c, gate) : gate,
                                                                gate_up_x ? ggml_cont(c, up) : up, 1.702f, 7.0f)
                              : ggml_swiglu_split(c, gate, up);   // one step (gate_up: on the two halves in place)
            ggml_tensor* down = ggml_mul_mat_id(c, down_x, act, ids);        // [n_embd, n_used, n_tok]
            if (down_b) down = ggml_add_id(c, down, down_b, bias_ids);
            down = ggml_mul(c, down, wt);                                    // weight each expert's output
            ggml_tensor* sum = nullptr;
            for (int e = 0; e < NU; ++e) {
                ggml_tensor* s = ggml_view_2d(c, down, XW, n_tok, down->nb[2], e * down->nb[1]);
                sum = sum ? ggml_add(c, sum, s) : s;
            }
            return sum;
        };
        ggml_tensor* moe;
        ggml_tensor* sh_early = nullptr;                                     // the shared expert, if made before the RAM experts
        auto shared_expert = [&]() {
            if (!L.gate_s) return ggml_mul_mat(c, L.down_s, ggml_sqr(c, ggml_relu(c, ggml_mul_mat(c, L.up_s, h))));   // Nemotron
            ggml_tensor* sh = ggml_swiglu_split(c, ggml_mul_mat(c, L.gate_s, h), ggml_mul_mat(c, L.up_s, h));
            sh = ggml_mul_mat(c, L.down_s, sh);                              // [n_embd, n_tok]
            if (L.gate_inp_s) sh = ggml_mul(c, sh, ggml_sigmoid(c, ggml_mul_mat(c, L.gate_inp_s, h)));   // gate: [1, n_tok]
            return sh;
        };
        if (L.map_hot) {
            // Hot-expert split: each pick is computed by the part that holds it (VRAM or RAM); in the other part it
            // goes to the all-zero expert with weight 0, so it adds nothing. The sum is the same as unsplit.
            // Looked up for the whole ranking (contiguous), then the top n_used rows are used, like sel.
            ggml_tensor* order1 = ggml_reshape_1d(c, order, (int64_t)NE * n_tok);
            auto lookup = [&](ggml_tensor* map) {                            // [n_used, n_tok] view of map[order]
                ggml_tensor* all = ggml_reshape_2d(c, ggml_get_rows(c, map, order1), NE, n_tok);
                return ggml_view_2d(c, all, NU, n_tok, all->nb[1], 0);
            };
            // Made contiguous on the CPU before they go anywhere: a strided view copied to the GPU as an input
            // gave CUDA bad ids (an illegal memory access on an RTX 3070).
            // A token's picks that live in the other part all point at the zero expert, so the same id can repeat
            // within one token. GPU kernels assume a token never picks an expert twice (CUDA's sorter then writes
            // past its buffer), so the repeats are moved to distinct unused experts of this part; their weight is
            // 0 there, so the sum does not change.
            // The RAM part needs it only for big batches (a prompt): those are handed to the GPU, which copies the
            // experts over (ggml's op offload, from 32 tokens). For a few tokens the CPU does it: there the zero-expert
            // picks become -1 and are skipped (see skip_zero_ids), so the CPU computes only the experts in RAM.
            auto remap = [&](ggml_tensor* map, const ggml_tensor* part, bool distinct) {
                ggml_tensor* ids = ggml_cont(c, lookup(map));
                g.cpu_ops.push_back(ids);
                const int64_t n_part = part == L.down_xc && L.cold_mapped ? -part->ne[2] : part->ne[2];
                return ggml_map_custom1(c, ids, distinct ? distinct_ids : skip_zero_ids, 1, (void*)(intptr_t)n_part);
            };
            // All the bookkeeping runs on the CPU, next to the RAM experts, so the work changes device only twice
            // per layer (to the CPU after the router, back to the GPU for the VRAM experts), as with whole layers.
            ggml_tensor* mcont  = ggml_cont(c, lookup(L.hot_mask));
            ggml_tensor* mask   = ggml_reshape_3d(c, mcont, 1, NU, n_tok);
            ggml_tensor* w_hot  = ggml_mul(c, w, mask);                     // [1, n_used, n_tok]
            ggml_tensor* w_cold = ggml_sub(c, w, w_hot);
            // Handing the RAM experts' work to the GPU means copying all of a layer's RAM experts over for the batch,
            // the same cost whatever its size, so it only pays for big batches (see cold_offload_min_): below that
            // the CPU computes just the picks that are in RAM, as it does for a reply's steps.
            static const int offload_batch = [] { const char* e = getenv("GGML_OP_OFFLOAD_MIN_BATCH"); return e ? atoi(e) : 32; }();
            const bool cold_gpu = n_tok >= std::max(offload_batch, cold_offload_min_);
            ggml_tensor* ids_c  = remap(L.map_cold, L.down_xc, cold_gpu);
            ggml_tensor* ids_h  = remap(L.map_hot, L.down_xh, true);
            // Biases (gpt-oss) aren't split: each part has all of them (RAM: the file's; VRAM: a copy) and adds them
            // by the router's own ids. A pick that lives in the other part gets a bias here too, on top of the zero
            // expert's zeros, and then its weight of 0, so it still adds nothing.
            ggml_tensor* ids_b = nullptr;
            if (L.up_xb || L.gate_xb || L.down_xb) { ids_b = ggml_cont(c, sel); g.cpu_ops.push_back(ids_b); }
            for (ggml_tensor* t : {mcont->src[0]->view_src, mcont, w_hot, w_cold, ids_c, ids_c->src[0]->src[0]->view_src, ids_h, ids_h->src[0]->src[0]->view_src})
                if (t) g.cpu_ops.push_back(t);
            // The router weights of the chosen experts (picked out of probs, renormalized, scaled) too.
            std::function<void(ggml_tensor*, int)> on_cpu = [&](ggml_tensor* t, int depth) {
                if (!t || t == probs || t == order || depth > 8) return;
                const bool view = t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
                if (!view && t->op != GGML_OP_NONE) g.cpu_ops.push_back(t);
                for (ggml_tensor* src : t->src) if (src && src != probs && src != order) on_cpu(src, depth + 1);
            };
            on_cpu(w, 0);
            ggml_build_forward_expand(g.gf, w_cold);                        // weights first, then the lookups: one hand-off
            ggml_tensor *cold, *hot;
            if (overlap_) {
                // CPU and GPU at once: all the bookkeeping first (CPU), then the VRAM experts and the shared expert
                // (GPU: the scheduler starts them and goes on), then the RAM experts (CPU, while the GPU works).
                // The sum is the same as below, in the same order, so the result doesn't change.
                // The RAM experts' input is copied to the CPU with the bookkeeping, so their own step has nothing
                // to fetch from the GPU and needn't wait for it.
                ggml_tensor* h_cpu = ggml_cont(c, h3);
                g.cpu_ops.push_back(h_cpu);
                for (ggml_tensor* t : {w_hot, ids_c, ids_h, h_cpu, ids_b}) if (t) ggml_build_forward_expand(g.gf, t);
                hot = experts(L.gate_up_xh, L.gate_xh, L.up_xh, L.down_xh, ids_h, w_hot, L.gate_xbh, L.up_xbh, L.down_xbh, ids_b);
                ggml_build_forward_expand(g.gf, hot);
                if (L.up_s) { sh_early = shared_expert(); ggml_build_forward_expand(g.gf, sh_early); }
                xin = h_cpu;
                cold = experts(L.gate_up_xc, L.gate_xc, L.up_xc, L.down_xc, ids_c, w_cold, L.gate_xb, L.up_xb, L.down_xb, ids_b);
                xin = h3;
            } else {
                cold = experts(L.gate_up_xc, L.gate_xc, L.up_xc, L.down_xc, ids_c, w_cold, L.gate_xb, L.up_xb, L.down_xb, ids_b);
                hot  = experts(L.gate_up_xh, L.gate_xh, L.up_xh, L.down_xh, ids_h, w_hot, L.gate_xbh, L.up_xbh, L.down_xbh, ids_b);
            }
            if (!cold_gpu && n_tok >= offload_batch) {                      // (the scheduler would hand it to the GPU: keep it on the CPU)
                std::set<ggml_tensor*> seen;
                std::function<void(ggml_tensor*)> pin = [&](ggml_tensor* t) {
                    if (!t || t == ids_c || t == ids_b || t == w_cold || t == h3 || t == xin || t->op == GGML_OP_NONE || !seen.insert(t).second) return;
                    g.cpu_ops.push_back(t);
                    for (ggml_tensor* src : t->src) pin(src);
                };
                pin(cold);
            }
            moe = ggml_add(c, cold, hot);                                   // cold first: the graph runs the RAM part, then the VRAM part
        } else {
            ggml_tensor* ids = sel;
            if (L.ssd && n_tok < 32) {                                       // experts on the drive: read the picks ahead
                ids = ggml_cont(c, sel);
                g.cpu_ops.push_back(ids);
                ids = ggml_map_custom1(c, ids, prefetch_experts, 1, (void*)L.ssd);
                g.cpu_ops.push_back(ids);
            }
            moe = experts(L.gate_up_x, L.gate_x, L.up_x, L.down_x, ids, w, L.gate_xb, L.up_xb, L.down_xb);
        }
        if (L.lat_up) moe = ggml_mul_mat(c, L.lat_up, moe);                  // back to the model's width
        if (L.up_s) moe = ggml_add(c, moe, sh_early ? sh_early : shared_expert());   // shared expert
        return moe;
}

// Builds the compute graph for n_tok new tokens placed after n_past cached ones.
// Only a description: the scheduler allocates its memory on the right device later.
// TIERLM_PROFILE=1: every 64 graphs of up to 8 tokens (replies, not prompts), prints where the time per graph
// went: preparing it, and per device the input copies and the work. Each split is waited for while this is on.
void Model::profile_step(int n_tok, double prep_ms, double compute_ms, double out_ms, double build_ms, double alloc_ms) {
    if (n_tok > 8) { ggml_backend_sched_profile(nullptr, nullptr, nullptr, nullptr, nullptr, true); prof_ = {}; return; }
    prof_.graphs++; prof_.tokens += n_tok; prof_.prep += prep_ms; prof_.compute += compute_ms; prof_.out += out_ms;
    prof_.build += build_ms; prof_.alloc += alloc_ms;
    if (prof_.graphs < 64) return;
    double copy[16], work[16]; int64_t splits[16], inputs[16], graphs = 0;
    ggml_backend_sched_profile(copy, work, splits, inputs, &graphs, true);
    const double n = (double)prof_.graphs;
    std::fprintf(stderr, "profile: %d graphs, %.1f tokens each: %.2f ms each = prepare %.2f (build %.2f, place %.2f) + compute %.2f + read out %.2f\n",
                 prof_.graphs, prof_.tokens / n, (prof_.prep + prof_.compute + prof_.out) / n, prof_.prep / n, prof_.build / n,
                 prof_.alloc / n, prof_.compute / n, prof_.out / n);
    std::vector<ggml_backend_t> bs(gpus_.begin(), gpus_.end());
    bs.push_back(cpu_);
    double accounted = 0;
    for (size_t i = 0; i < bs.size() && i < 16; ++i)
        if (splits[i]) {
            std::fprintf(stderr, "profile:   %-10s %5.1f splits, %6.1f inputs: copy in %6.2f ms, work %6.2f ms per graph\n",
                         ggml_backend_name(bs[i]), splits[i] / n, inputs[i] / n, copy[i] / 1000.0 / n, work[i] / 1000.0 / n);
            accounted += (copy[i] + work[i]) / 1000.0 / n;
        }
    // what's left of the compute time: starting each split, waiting for it, switching devices
    std::fprintf(stderr, "profile:   between splits (starting, waiting, switching) %.2f ms per graph; CPU threads %d\n",
                 std::max(0.0, prof_.compute / n - accounted), n_threads_last_);
    prof_ = {};
}

Model::Graph Model::build(int n_tok, int n_kv_end, bool want_routing, int n_logits, bool want_hidden, int snapshots,
                          const std::vector<SeqGroup>* groups, std::vector<uint8_t>* mem) {
    const Hparams& hp = hp_;
    const int n_kv = kv_span(n_kv_end), n_past = n_kv_end - n_tok;   // (attn_block takes the span as n_past + n_tok)
    const SeqGroup one{0, n_tok};
    const std::vector<SeqGroup> single(1, one);
    const std::vector<SeqGroup>& grp = groups ? *groups : single;
    const int hd = hp.head_dim;

    if (!mem) drop_reuse();                                        // (its memory is about to be built over)
    Graph g;
    std::vector<uint8_t>& gm = mem ? *mem : graph_mem(n_tok == 1 ? 0 : 1, n_tok);
    ggml_init_params ip{ gm.size(), gm.data(), /*no_alloc=*/true };
    ggml_context* c = g.ctx = ggml_init(ip);
    g.gf = ggml_new_graph_custom(c, graph_size_, false);

    g.inp = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tok);
    ggml_set_input(g.inp);
    g.mrope = build_mrope_;
    g.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, g.mrope ? 4 * n_tok : n_tok);
    ggml_set_input(g.pos);
    if (build_images_) {
        g.img = ggml_new_tensor_2d(c, GGML_TYPE_F32, hp.n_embd, n_tok);  ggml_set_input(g.img);
        g.keep = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, n_tok);         ggml_set_input(g.keep);
    }
    g.cells = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tok);
    ggml_set_input(g.cells);
    if (hp.mamba) { g.zero_id = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1); ggml_set_input(g.zero_id); }
    // Mask (0 = visible, -inf = not): each token sees its own conversation's earlier positions (and itself).
    g.mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv, n_tok);
    ggml_set_input(g.mask);
    if (hp.n_swa > 0) {                                                    // the same, limited to the last n_swa tokens
        g.mask_swa = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv, n_tok);
        ggml_set_input(g.mask_swa);
    }
    if (!hp.rope_ff.empty()) {
        g.rope_ff = ggml_new_tensor_1d(c, GGML_TYPE_F32, (int64_t)hp.rope_ff.size());
        ggml_set_input(g.rope_ff);
    }
    if (ple_embd_) {                                                       // Flash-Next: the n-gram rows of every token
        g.ngram = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t)hp.ngram.heads() * n_tok);
        ggml_set_input(g.ngram);
    }
    const float scale = 1.0f / std::sqrt((float)hd);

    ggml_tensor* x = ggml_get_rows(c, tok_embd_, g.inp);                   // [n_embd, n_tok]
    if (g.img) x = ggml_add(c, ggml_mul(c, x, g.keep), g.img);             // images' words in place of their tokens
    // ---- Flash-Next's hyper-connections (Hugging Face's Qwen4ExpTextGatedResidual) ----
    // The residual stream is HC copies of the width, [HC * E, n_tok], starting as HC copies of the embedding. Before
    // each block, the copies are normed one by one (hc_norm), low-rank mixing weights are made from all of them,
    // sigmoid(up(silu(down(normed) / HC))), and the block's input is the mean over the copies of weights x normed.
    // The block's output is added back to every copy, times 2 sigmoid(inject(normed) / HC) for that copy.
    const int HC = hp.hc, E = hp.n_embd;
    // TIERLM_TRACE=1 (debugging): name some of each layer's results so step() prints a summary of each one
    static const bool trace = std::getenv("TIERLM_TRACE") != nullptr;
    auto tr = [&](ggml_tensor* t, const char* what, int layer) {
        if (trace && t) ggml_format_name(t, "tr:%s-%d", what, layer);
        return t;
    };
    auto gnorm = [&](ggml_tensor* t, ggml_tensor* w, int64_t T) {          // RMS norm over each copy, times w
        ggml_tensor* r = ggml_rms_norm(c, ggml_reshape_3d(c, t, E, HC, T), hp.rms_eps);
        return ggml_mul(c, ggml_reshape_2d(c, r, (int64_t)E * HC, T), w);
    };
    auto hc_mix = [&](ggml_tensor* X, ggml_tensor* nw, ggml_tensor* down, ggml_tensor* up, ggml_tensor* injw, ggml_tensor** inj) {
        ggml_tensor* xn = gnorm(X, nw, n_tok);
        ggml_tensor* m = ggml_sigmoid(c, ggml_mul_mat(c, up, ggml_silu(c, ggml_scale(c, ggml_mul_mat(c, down, xn), 1.0f / HC))));
        ggml_tensor* p = ggml_mul(c, m, xn);                               // [HC * E, n_tok]
        ggml_tensor* s = nullptr;
        for (int k = 0; k < HC; ++k) {
            ggml_tensor* v = ggml_view_2d(c, p, E, n_tok, p->nb[1], (size_t)k * E * ggml_element_size(p));
            s = s ? ggml_add(c, s, v) : v;
        }
        s = ggml_scale(c, s, 1.0f / HC);                                   // [E, n_tok]
        if (inj) *inj = ggml_scale(c, ggml_sigmoid(c, ggml_scale(c, ggml_mul_mat(c, injw, xn), 1.0f / HC)), 2.0f);   // [HC, n_tok]
        return s;
    };
    auto hc_add = [&](ggml_tensor* X, ggml_tensor* y, ggml_tensor* inj) {  // X + y into every copy, times inj
        ggml_tensor* y3 = ggml_reshape_3d(c, ggml_cont(c, y), E, 1, n_tok);
        ggml_tensor* r = ggml_mul(c, ggml_repeat_4d(c, y3, E, HC, n_tok, 1), ggml_reshape_3d(c, inj, 1, HC, n_tok));
        return ggml_add(c, X, ggml_reshape_2d(c, r, (int64_t)E * HC, n_tok));
    };
    // ---- Flash-Next's per-layer embedding (Hugging Face's Qwen4ExpTextPLELayer) ----
    // The token's n-gram rows (heads x ple_dim numbers) give one value for the width and a key for every copy;
    // each copy's normed activations gate the value: g = sum(key x query) / sqrt(E), sign(g) sqrt(|g|), sigmoid.
    // The gated values, normed, go through a dilated causal convolution (taps ngram_size apart) with silu, added
    // to the gated values. The convolution's last inputs are kept per conversation (Layer::ple_states).
    auto ple = [&](const Layer& L, ggml_tensor* X) -> ggml_tensor* {
        const int NH = hp.ngram.heads(), D = hp.ple_dim, K = hp.ple_kernel, DIL = hp.ngram.ngram, SL = hp.ple_state_len();
        const int64_t C = (int64_t)HC * E;
        ggml_tensor* emb = ggml_reshape_2d(c, ggml_get_rows(c, ple_embd_, g.ngram), (int64_t)D * NH, n_tok);
        ggml_tensor* key = gnorm(ggml_mul_mat(c, L.ple_key, emb), L.ple_norm_key, n_tok);   // [C, n_tok]
        ggml_tensor* val = ggml_mul_mat(c, L.ple_value, emb);                               // [E, n_tok]
        ggml_tensor* qry = gnorm(X, L.ple_norm_query, n_tok);
        ggml_tensor* gt = ggml_sum_rows(c, ggml_reshape_3d(c, ggml_mul(c, key, qry), E, HC, n_tok));   // [1, HC, n_tok]
        gt = ggml_scale(c, gt, 1.0f / std::sqrt((float)E));
        gt = ggml_mul(c, ggml_sgn(c, gt), ggml_sqrt(c, ggml_clamp(c, ggml_abs(c, gt), 1e-6f, 3.0e38f)));
        ggml_tensor* gv = ggml_mul(c, ggml_repeat_4d(c, ggml_reshape_3d(c, val, E, 1, n_tok), E, HC, n_tok, 1), ggml_sigmoid(c, gt));
        gv = ggml_reshape_2d(c, gv, C, n_tok);                                               // [C, n_tok]
        ggml_tensor* gvn = gnorm(gv, L.ple_norm_conv, n_tok);
        // the taps, one [C, 1] each (channel c's taps are K numbers in a row)
        std::vector<ggml_tensor*> w(K);
        for (int k = 0; k < K; ++k) {
            ggml_tensor* v = ggml_cont(c, ggml_view_2d(c, L.ple_conv, 1, C, L.ple_conv->nb[1], (size_t)k * ggml_element_size(L.ple_conv)));
            if (v->type != GGML_TYPE_F32) v = ggml_cast(c, v, GGML_TYPE_F32);
            w[k] = ggml_reshape_2d(c, v, C, 1);
        }
        ggml_tensor* co = nullptr;
        for (int gi = 0, t0 = 0; gi < (int)grp.size(); t0 += grp[gi].n, ++gi) {
            const int n = grp[gi].n;
            ggml_tensor* st = L.ple_states[grp[gi].seq];                                     // [C, SL]
            ggml_tensor* cur = n == n_tok ? gvn : ggml_view_2d(c, gvn, C, n, gvn->nb[1], (size_t)t0 * gvn->nb[1]);
            ggml_tensor* xc = ggml_concat(c, st, cur, 1);                                    // [C, SL + n]
            ggml_build_forward_expand(g.gf, ggml_cpy(c, ggml_view_2d(c, xc, C, SL, xc->nb[1], (size_t)n * xc->nb[1]), st));
            if (snapshots > 1) {
                // For dropping the last d tokens: the inputs kept as they were before them (as DeltaNet's rollback).
                auto& rb = g.ple_rollback[(int)(&L - layers_.data())];
                rb.assign(std::min(snapshots, n), nullptr);
                for (int d = 1; d < (int)rb.size(); ++d) {
                    rb[d] = ggml_cont(c, ggml_view_2d(c, xc, C, SL, xc->nb[1], (size_t)(n - d) * xc->nb[1]));
                    ggml_set_output(rb[d]);
                    ggml_build_forward_expand(g.gf, rb[d]);
                }
            }
            ggml_tensor* sum = nullptr;
            for (int k = 0; k < K; ++k) {
                ggml_tensor* tap = ggml_mul(c, ggml_view_2d(c, xc, C, n, xc->nb[1], (size_t)k * DIL * xc->nb[1]), w[k]);
                sum = sum ? ggml_add(c, sum, tap) : tap;
            }
            sum = ggml_silu(c, sum);
            co = co ? ggml_concat(c, co, sum, 1) : sum;
        }
        return ggml_add(c, gv, co);
    };
    if (HC) x = ggml_reshape_2d(c, ggml_repeat_4d(c, ggml_reshape_3d(c, x, E, 1, n_tok), E, HC, n_tok, 1), (int64_t)E * HC, n_tok);
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
        ggml_tensor* inj = nullptr;                                          // (Flash-Next: each copy's share)
        if (li == 0) tr(x, "in", -1);
        if (HC && L.ple_key) x = ggml_add(c, x, tr(ple(L, x), "ple", li));
        ggml_tensor* h = HC ? hc_mix(x, L.hc_a_norm, L.hc_a_down, L.hc_a_up, L.hc_a_inj, &inj)
                            : ggml_mul(c, ggml_rms_norm(c, x, hp.rms_eps), L.attn_norm);
        tr(h, "mix_attn", li); tr(inj, "inj_attn", li);
        auto add_back = [&](ggml_tensor* d) { x = HC ? hc_add(x, d, inj) : ggml_add(c, x, d); };
        if (hp.mamba && L.kind == Hparams::kMamba) {
            // ---- Mamba-2 (Nemotron's running memory) ----
            // One projection gives, per token: z (the output gate), x with B and C (through a short causal
            // convolution and silu), and dt (a step size per head). Each head keeps a state [head width, m_state]:
            //   state = state * exp(dt * A) + dt * x (outer) B,   y = state . C + D * x
            // with dt = softplus(dt + bias), A < 0 per head, and each group's B and C shared by its heads. Then
            // y * silu(z), normed group by group, and back to the model's width.
            const int DI = hp.m_inner, MH = hp.m_heads, DS = hp.m_state, MG = hp.m_groups, KC = hp.ssm_conv;
            const int HDM = DI / MH, CC = hp.ssm_conv_dim();
            ggml_tensor* zx = ggml_mul_mat(c, L.ssm_in, h);                       // [2 DI + 2 MG DS + MH, n_tok]
            if (L.ssm_in_s) zx = ggml_mul(c, zx, L.ssm_in_s);                     // (4-bit files: a scale beside the weights)
            const size_t ez = ggml_element_size(zx);
            auto run = [&](int sq, int t0, int n) -> ggml_tensor* {
                auto rows = [&](int64_t w, int64_t off) {                         // rows [off, off + w) of this conversation's tokens
                    return ggml_cont(c, ggml_view_2d(c, zx, w, n, zx->nb[1], (size_t)t0 * zx->nb[1] + (size_t)off * ez));
                };
                ggml_tensor* conv_state = L.conv_states[sq];
                ggml_tensor* ssm_state  = L.ssm_states[sq];
                ggml_tensor* z = rows(DI, 0);
                // Causal convolution over the last KC inputs: [the KC-1 saved from before | this batch]
                ggml_tensor* cs  = ggml_reshape_3d(c, conv_state, KC - 1, CC, 1);
                ggml_tensor* cin = ggml_concat(c, cs, ggml_transpose(c, ggml_reshape_3d(c, rows(CC, DI), CC, n, 1)), 0);
                ggml_build_forward_expand(g.gf, ggml_cpy(c, ggml_view_3d(c, cin, KC - 1, CC, 1, cin->nb[1], cin->nb[2],
                                                                         (size_t)n * ggml_element_size(cin)), cs));
                ggml_tensor* co = ggml_ssm_conv(c, cin, L.ssm_conv1d);            // [CC, n, 1]
                if (L.ssm_conv_b) co = ggml_add(c, co, L.ssm_conv_b);
                co = ggml_silu(c, co);
                const size_t es = ggml_element_size(co), rs = co->nb[1];
                ggml_tensor* xh = ggml_view_4d(c, co, HDM, MH, n, 1, HDM * es, rs, rs * n, 0);
                ggml_tensor* B  = ggml_view_4d(c, co, DS, MG, n, 1, DS * es, rs, rs * n, (size_t)DI * es);
                ggml_tensor* C  = ggml_view_4d(c, co, DS, MG, n, 1, DS * es, rs, rs * n, (size_t)(DI + MG * DS) * es);
                ggml_tensor* dtb = ggml_add(c, rows(MH, DI + CC), L.ssm_dt);     // [MH, n]: before the softplus
                ggml_tensor* y = nullptr;
                std::vector<ggml_tensor*> back;                                   // the state as it was d tokens back (d >= 1)
                // ggml's scan runs on the CPU and on CUDA (where it also keeps the undo points of a check of guesses:
                // 139 tok/s on an RTX 5090 against 114 written out). Vulkan's scan doesn't keep undo points, so on a
                // GPU other than CUDA a check of guesses has the recurrence written out with ordinary operations,
                // token by token, which keeps the states between the tokens.
                static const int unroll_env = [] { const char* e = std::getenv("TIERLM_MAMBA_UNROLL"); return e ? std::atoi(e) : -1; }();
                const int dev = li < (int)layer_dev_.size() ? layer_dev_[li] : -1;
                const bool cuda_dev = dev >= 0 && dev < (int)gpus_.size() &&
                                      std::strncmp(ggml_backend_dev_name(ggml_backend_get_device(gpus_[dev])), "CUDA", 4) == 0;
                const bool unroll = unroll_env >= 0 ? n <= unroll_env : snapshots > 1 && dev >= 0 && !cuda_dev;
                if (unroll) {
                    const int R = MH / MG;
                    ggml_tensor* dts = ggml_softplus(c, dtb);                                           // [MH, n]
                    ggml_tensor* dA  = ggml_exp(c, ggml_mul(c, dts, ggml_reshape_2d(c, L.ssm_a, MH, 1)));
                    ggml_tensor* xdt = ggml_mul(c, ggml_reshape_3d(c, ggml_cont(c, xh), HDM, MH, n), ggml_reshape_3d(c, dts, 1, MH, n));
                    auto heads = [&](ggml_tensor* t) {                             // each group's B or C for its R heads in a row
                        t = ggml_repeat_4d(c, ggml_reshape_4d(c, ggml_cont(c, t), DS, 1, MG, n), DS, R, MG, n);
                        return ggml_reshape_3d(c, t, DS, MH, n);
                    };
                    ggml_tensor *Bh = heads(B), *Ch = heads(C);
                    ggml_tensor* S = ssm_state;                                                          // [DS, HDM, MH]
                    std::vector<ggml_tensor*> states;
                    for (int t = 0; t < n; ++t) {
                        auto col = [&](ggml_tensor* a, int64_t d0, int64_t d1, int64_t d2) {             // token t of [.., .., n]
                            const size_t per = (size_t)(ggml_nelements(a) / n) * ggml_element_size(a);
                            return ggml_reshape_3d(c, ggml_view_1d(c, a, ggml_nelements(a) / n, (size_t)t * per), d0, d1, d2);
                        };
                        ggml_tensor* outer = ggml_mul(c, ggml_repeat_4d(c, col(Bh, DS, 1, MH), DS, HDM, MH, 1), col(xdt, 1, HDM, MH));
                        S = ggml_add(c, ggml_mul(c, S, col(dA, 1, 1, MH)), outer);
                        states.push_back(S);
                        ggml_tensor* yt = ggml_reshape_2d(c, ggml_sum_rows(c, ggml_mul(c, S, col(Ch, DS, 1, MH))), DI, 1);
                        y = y ? ggml_concat(c, y, yt, 1) : yt;                                           // [DI, t + 1]
                    }
                    y = ggml_reshape_4d(c, y, HDM, MH, n, 1);
                    ggml_build_forward_expand(g.gf, ggml_cpy(c, S, ssm_state));
                    for (int d = 1; d < std::min(snapshots, n); ++d) back.push_back(ggml_cont(c, states[n - 1 - d]));
                } else {
                    ggml_tensor* st = ggml_reshape_4d(c, ssm_state, DS, HDM, MH, 1);
                    // Outputs [HDM, MH, n], then the new state. With snapshots > 1 the scan also keeps the state as it
                    // was 1, 2, ... tokens back, so a speculative check can be undone for the tokens that weren't accepted.
                    ggml_tensor* res = ggml_ssm_scan(c, st, xh, ggml_reshape_3d(c, dtb, MH, n, 1), L.ssm_a, B, C, g.zero_id,
                                                     std::max(1, std::min(snapshots, n)));       // (the scan does the softplus)
                    const size_t y_bytes = (size_t)DI * n * 4, st_bytes = (size_t)DS * DI * 4;
                    y = ggml_view_4d(c, res, HDM, MH, n, 1, (size_t)HDM * 4, (size_t)DI * 4, y_bytes, 0);
                    ggml_tensor* ns = ggml_view_1d(c, res, (int64_t)DS * DI, y_bytes);
                    ggml_build_forward_expand(g.gf, ggml_cpy(c, ns, ggml_view_1d(c, ssm_state, (int64_t)DS * DI, 0)));
                    if (snapshots > 1) ggml_set_output(res);
                    for (int d = 1; d < std::min(snapshots, n); ++d)
                        back.push_back(ggml_view_3d(c, res, DS, HDM, MH, (size_t)DS * 4, (size_t)DS * HDM * 4, y_bytes + (size_t)d * st_bytes));
                }
                if (snapshots > 1) {
                    auto& rb = g.rollback[li];
                    rb.resize(std::min(snapshots, n));
                    for (int d = 1; d < (int)rb.size(); ++d) {
                        rb[d].first = back[d - 1];
                        rb[d].second = ggml_cont(c, ggml_view_3d(c, cin, KC - 1, CC, 1, cin->nb[1], cin->nb[2],
                                                                 (size_t)(n - d) * ggml_element_size(cin)));
                        if (unroll) ggml_set_output(rb[d].first);                 // (the scan's: views of its one result)
                        ggml_set_output(rb[d].second);
                        ggml_build_forward_expand(g.gf, rb[d].first);
                        ggml_build_forward_expand(g.gf, rb[d].second);
                    }
                }
                y = ggml_add(c, y, ggml_mul(c, xh, L.ssm_d));                     // D: each head's own share of x
                y = ggml_mul(c, ggml_reshape_2d(c, ggml_cont(c, y), DI, n), ggml_silu(c, z));
                if (L.ssm_norm) {                                                 // each group normed on its own
                    y = ggml_mul(c, ggml_rms_norm(c, ggml_reshape_3d(c, y, DI / MG, MG, n), hp.rms_eps), L.ssm_norm);
                    y = ggml_reshape_2d(c, y, DI, n);
                }
                return y;
            };
            ggml_tensor* y = nullptr;
            for (int gi = 0, t0 = 0; gi < (int)grp.size(); t0 += grp[gi].n, ++gi) {
                ggml_tensor* part = run(grp[gi].seq, t0, grp[gi].n);
                y = y ? ggml_concat(c, y, part, 1) : part;
            }
            ggml_tensor* o = ggml_mul_mat(c, L.ssm_out, y);
            if (L.ssm_out_s) o = ggml_mul(c, o, L.ssm_out_s);
            add_back(tr(o, "mamba_out", li));
            continue;
        }
        if (hp.mamba) {                                                           // attention alone, or the experts alone
            add_back(L.kind == Hparams::kAttn ? tr(attn_block(c, g, L, h, n_tok, n_past), "attn_out", li)
                                              : tr(ffn_block(c, g, L, h, n_tok, want_routing), "ffn_out", li));
            continue;
        }
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
            // One conversation's tokens [t0, t0 + n) through the convolution and the memory, with that
            // conversation's saved inputs and memory; returns the gated, normed output [S*Hv, n].
            auto run = [&](int sq, int t0, int n) -> ggml_tensor* {
                const bool all = n == n_tok;
                auto cols = [&](ggml_tensor* t) {                                 // columns [t0, t0 + n) of [rows, n_tok]
                    return all ? t : ggml_view_2d(c, t, t->ne[0], n, t->nb[1], (size_t)t0 * t->nb[1]);
                };
                ggml_tensor* conv_state = L.conv_states[sq];
                ggml_tensor* ssm_state  = L.ssm_states[sq];
                ggml_tensor* beta = ggml_sigmoid(c, cols(b_in));                     // how strongly to write, per head
                beta = ggml_reshape_4d(c, beta, 1, Hv, n, 1);
                // log of the memory's decay: -exp(A_log) * softplus(alpha + dt_bias)  (ssm_a holds -exp(A_log))
                ggml_tensor* dec = ggml_mul(c, ggml_softplus(c, ggml_add(c, cols(a_in), L.ssm_dt)), L.ssm_a);
                dec = ggml_reshape_4d(c, dec, 1, Hv, n, 1);
                // Causal convolution over the last KC inputs: [the KC-1 saved from before | this batch]
                ggml_tensor* cs  = ggml_reshape_3d(c, conv_state, KC - 1, CD, 1);
                ggml_tensor* q_in = all ? qkv : ggml_cont(c, cols(qkv));
                ggml_tensor* cin = ggml_concat(c, cs, ggml_transpose(c, ggml_reshape_3d(c, q_in, CD, n, 1)), 0);
                ggml_build_forward_expand(g.gf, ggml_cpy(c, ggml_view_3d(c, cin, KC - 1, CD, 1, cin->nb[1], cin->nb[2],
                                                                         (size_t)n * ggml_element_size(cin)), cs));
                ggml_tensor* co = ggml_silu(c, ggml_ssm_conv(c, cin, L.ssm_conv1d)); // [CD, n, 1]
                tr(q_in, "dn_qkv", li); tr(co, "dn_conv", li); tr(dec, "dn_decay", li); tr(beta, "dn_beta", li);
                const size_t es = ggml_element_size(co), rs = co->nb[1];
                ggml_tensor* qh = ggml_view_4d(c, co, S, Hk, n, 1, S * es, rs, rs * n, 0);
                ggml_tensor* kh = ggml_view_4d(c, co, S, Hk, n, 1, S * es, rs, rs * n, (size_t)S * Hk * es);
                ggml_tensor* vh = ggml_view_4d(c, co, S, Hv, n, 1, S * es, rs, rs * n, (size_t)2 * S * Hk * es);
                // l2-normalise q and k: x / sqrt(sum(x^2) + eps), written with rms_norm so every backend has it
                auto l2 = [&](ggml_tensor* t) { return ggml_scale(c, ggml_rms_norm(c, t, hp.rms_eps / S), 1.0f / std::sqrt((float)S)); };
                qh = l2(qh);
                kh = l2(kh);
                tr(qh, "dn_q", li); tr(kh, "dn_k", li);
                if (hp.v_grouped && R > 1) {
                    // Value heads grouped by key head (head j uses key head j / R): repeat each key head R times in place.
                    auto rep_heads = [&](ggml_tensor* t) {
                        t = ggml_repeat_4d(c, ggml_reshape_4d(c, t, S, 1, Hk, n), S, R, Hk, n);
                        return ggml_reshape_4d(c, t, S, Hv, n, 1);
                    };
                    qh = rep_heads(qh);
                    kh = rep_heads(kh);
                }
                ggml_tensor* st  = ggml_reshape_4d(c, ssm_state, S, S, Hv, 1);
                // Outputs, then the new memory. With snapshots > 1 the op also keeps the memory as it was 1, 2, ...
                // tokens back, so a speculative check can be undone for the tokens that weren't accepted.
                ggml_tensor* res = ggml_gated_delta_net(c, qh, kh, vh, dec, beta, st, snapshots);
                ggml_tensor* out = ggml_view_4d(c, res, S, Hv, n, 1, (size_t)S * 4, (size_t)S * Hv * 4, (size_t)S * Hv * n * 4, 0);
                ggml_tensor* ns  = ggml_view_1d(c, res, (int64_t)S * S * Hv, (size_t)S * Hv * n * 4);
                ggml_build_forward_expand(g.gf, ggml_cpy(c, ns, ggml_view_1d(c, ssm_state, (int64_t)S * S * Hv, 0)));
                if (snapshots > 1) {
                    // For dropping the last d tokens: the memory d tokens back, and the convolution's last inputs before
                    // them. Kept as outputs until Model::accept() has used them. (One conversation only.)
                    ggml_set_output(res);
                    auto& rb = g.rollback[li];
                    rb.resize(std::min(snapshots, n));
                    for (int d = 1; d < (int)rb.size(); ++d) {
                        rb[d].first = ggml_view_3d(c, res, S, S, Hv, (size_t)S * 4, (size_t)S * S * 4,
                                                   ((size_t)S * Hv * n + (size_t)d * S * S * Hv) * 4);
                        rb[d].second = ggml_cont(c, ggml_view_3d(c, cin, KC - 1, CD, 1, cin->nb[1], cin->nb[2],
                                                                 (size_t)(n - d) * ggml_element_size(cin)));
                        ggml_set_output(rb[d].second);
                        ggml_build_forward_expand(g.gf, rb[d].first);      // views get their memory only as graph nodes
                        ggml_build_forward_expand(g.gf, rb[d].second);
                    }
                }
                // Gated norm: rms_norm(out) * w * silu(z) (Flash-Next: sigmoid(z)), then back to the model width
                ggml_tensor* z_in = all ? z : ggml_cont(c, cols(z));
                tr(out, "dn_out", li); tr(z_in, "dn_z", li);
                ggml_tensor* z4 = ggml_reshape_4d(c, z_in, S, Hv, n, 1);
                ggml_tensor* on = ggml_mul(c, ggml_mul(c, ggml_rms_norm(c, out, hp.rms_eps), L.ssm_norm),
                                           hp.dn_sigmoid_gate ? ggml_sigmoid(c, z4) : ggml_silu(c, z4));
                return ggml_reshape_2d(c, on, (int64_t)S * Hv, n);
            };
            ggml_tensor* on = nullptr;
            for (int gi = 0, t0 = 0; gi < (int)grp.size(); t0 += grp[gi].n, ++gi) {
                ggml_tensor* part = run(grp[gi].seq, t0, grp[gi].n);
                on = on ? ggml_concat(c, on, part, 1) : part;
            }
            add_back(tr(ggml_mul_mat(c, L.ssm_out, on), "attn_out", li));
        } else {
            add_back(tr(attn_block(c, g, L, h, n_tok, n_past), "attn_out", li));
        }
        tr(x, "res_attn", li);

        h = HC ? hc_mix(x, L.hc_f_norm, L.hc_f_down, L.hc_f_up, L.hc_f_inj, &inj) : ggml_mul(c, ggml_rms_norm(c, x, hp.rms_eps), L.ffn_norm);
        tr(h, "mix_ffn", li); tr(inj, "inj_ffn", li);
        add_back(tr(ffn_block(c, g, L, h, n_tok, want_routing), "ffn_out", li));
        tr(x, "res", li);
    }
    tag((int)layers_.size() - 1);

    if (!g.orders.empty()) {                                               // every layer's picks in one tensor
        // (Nemotron: blocks pick different numbers of experts, so the lists are joined end to end instead)
        bool even = true;
        for (ggml_tensor* o : g.orders) even &= o->ne[0] == g.orders[0]->ne[0];
        ggml_tensor* r = even ? g.orders[0] : ggml_reshape_1d(c, g.orders[0], ggml_nelements(g.orders[0]));
        for (size_t l = 1; l < g.orders.size(); ++l)
            r = even ? ggml_concat(c, r, g.orders[l], 1)                                       // [n_used, n_tok * L]
                     : ggml_concat(c, r, ggml_reshape_1d(c, g.orders[l], ggml_nelements(g.orders[l])), 0);
        ggml_set_output(r);
        ggml_build_forward_expand(g.gf, r);
        g.routing = r;
    }

    if (want_hidden && HC) {                                                // [HC * E, n_tok]: the copies, for the MTP block
        g.hidden = ggml_cont(c, x); ggml_set_output(g.hidden); ggml_build_forward_expand(g.gf, g.hidden);
    }
    x = HC ? ggml_cont(c, hc_mix(x, out_hc_norm_, out_hc_down_, out_hc_up_, nullptr, nullptr))   // the last mixer
           : ggml_mul(c, ggml_rms_norm(c, x, hp.rms_eps), out_norm_);
    tr(x, "final", -1);
    if (want_hidden && !HC) { g.hidden = x; ggml_set_output(x); }          // [n_embd, n_tok]: what the MTP head reads
    ggml_tensor* last;
    if (groups) {                                                           // several conversations: each one's last token
        g.out_idx = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t)grp.size());
        ggml_set_input(g.out_idx);
        last = ggml_get_rows(c, x, g.out_idx);                              // [n_embd, n_groups]
    } else {
        n_logits = std::max(1, std::min(n_logits, n_tok));                 // predictions for the last n_logits positions
        last = ggml_view_2d(c, x, hp.n_embd, n_logits, x->nb[1], (size_t)(n_tok - n_logits) * x->nb[1]);
    }
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
Model::Graph Model::build_mtp(int n_tok, int n_past, std::vector<uint8_t>* mem) {
    const Hparams& hp = hp_;
    const Layer& L = *mtp_;
    const int n_kv = kv_span(n_past + n_tok);
    Graph g;
    std::vector<uint8_t>& gm = mem ? *mem : graph_mem(2, n_tok);
    ggml_init_params ip{ gm.size(), gm.data(), /*no_alloc=*/true };
    ggml_context* c = g.ctx = ggml_init(ip);
    g.gf = ggml_new_graph_custom(c, kGraphSize, false);
    g.n_tok = n_tok;
    g.inp = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tok);  ggml_set_input(g.inp);
    g.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tok);  ggml_set_input(g.pos);
    g.h_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, hp.hidden_width(), n_tok);  ggml_set_input(g.h_in);
    if (flash_attn_) { g.mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv, n_tok); ggml_set_input(g.mask); }
    if (hp.hc) {
        // Flash-Next: the model's hc copies (before its last mixer) are normed one by one; each copy, beside the
        // normed embedding, goes through eh_proj, which gives the block's own hc copies. Then one full-attention
        // block with its stream mixers, as in the model, and the block's own last mixer before the output head.
        // Its copies stand in for the model's, to draft further ahead.
        const int HC = hp.hc, E = hp.n_embd;
        auto gnorm = [&](ggml_tensor* t, ggml_tensor* w) {                  // RMS norm over each copy, times w
            ggml_tensor* r = ggml_rms_norm(c, ggml_reshape_3d(c, t, E, HC, n_tok), hp.rms_eps);
            return ggml_mul(c, ggml_reshape_2d(c, r, (int64_t)E * HC, n_tok), w);
        };
        auto mix = [&](ggml_tensor* X, ggml_tensor* nw, ggml_tensor* down, ggml_tensor* up, ggml_tensor* injw, ggml_tensor** inj) {
            ggml_tensor* xn = gnorm(X, nw);
            ggml_tensor* mw = ggml_sigmoid(c, ggml_mul_mat(c, up, ggml_silu(c, ggml_scale(c, ggml_mul_mat(c, down, xn), 1.0f / HC))));
            ggml_tensor* p = ggml_mul(c, mw, xn);
            ggml_tensor* s = nullptr;
            for (int k = 0; k < HC; ++k) {
                ggml_tensor* v = ggml_view_2d(c, p, E, n_tok, p->nb[1], (size_t)k * E * ggml_element_size(p));
                s = s ? ggml_add(c, s, v) : v;
            }
            s = ggml_scale(c, s, 1.0f / HC);
            if (inj) *inj = ggml_scale(c, ggml_sigmoid(c, ggml_scale(c, ggml_mul_mat(c, injw, xn), 1.0f / HC)), 2.0f);
            return s;
        };
        auto add = [&](ggml_tensor* X, ggml_tensor* y, ggml_tensor* inj) {
            ggml_tensor* y3 = ggml_reshape_3d(c, ggml_cont(c, y), E, 1, n_tok);
            ggml_tensor* r = ggml_mul(c, ggml_repeat_4d(c, y3, E, HC, n_tok, 1), ggml_reshape_3d(c, inj, 1, HC, n_tok));
            return ggml_add(c, X, ggml_reshape_2d(c, r, (int64_t)E * HC, n_tok));
        };
        ggml_tensor* e = ggml_get_rows(c, L.mtp_embd ? L.mtp_embd : tok_embd_, g.inp);
        e = ggml_mul(c, ggml_rms_norm(c, e, hp.rms_eps), L.enorm);                                   // [E, n_tok]
        ggml_tensor* e3 = ggml_repeat_4d(c, ggml_reshape_3d(c, e, E, 1, n_tok), E, HC, n_tok, 1);   // [E, HC, n_tok]
        ggml_tensor* hn = ggml_reshape_3d(c, gnorm(g.h_in, L.hnorm), E, HC, n_tok);
        ggml_tensor* x = ggml_mul_mat(c, L.eh_proj, ggml_concat(c, e3, hn, 0));                      // [E, HC, n_tok]
        x = ggml_reshape_2d(c, x, (int64_t)E * HC, n_tok);
        ggml_tensor* inj = nullptr;
        ggml_tensor* h = mix(x, L.hc_a_norm, L.hc_a_down, L.hc_a_up, L.hc_a_inj, &inj);
        x = add(x, attn_block(c, g, L, h, n_tok, n_past), inj);
        h = mix(x, L.hc_f_norm, L.hc_f_down, L.hc_f_up, L.hc_f_inj, &inj);
        x = add(x, ffn_block(c, g, L, h, n_tok, false), inj);
        g.hidden = ggml_cont(c, x); ggml_set_output(g.hidden); ggml_build_forward_expand(g.gf, g.hidden);   // [HC * E, n_tok]
        ggml_tensor* o = ggml_cont(c, mix(x, L.hc_head_norm, L.hc_head_down, L.hc_head_up, nullptr, nullptr));   // [E, n_tok]
        ggml_tensor* last = ggml_view_2d(c, o, E, 1, o->nb[1], (size_t)(n_tok - 1) * o->nb[1]);
        g.logits = ggml_mul_mat(c, L.head ? L.head : output_, last);
        ggml_set_output(g.logits);
        ggml_build_forward_expand(g.gf, g.logits);
        return g;
    }

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
    std::lock_guard<std::recursive_mutex> dev_lock(device_mutex());   // another loaded model waits for this step
    if (pos0 + n_tok > max_ctx_) throw std::runtime_error("MTP: past the end of the context");
    release_pending();
    const Hparams& hp = hp_;
    // One token (each guess, and catching up after a check that kept none): the kept plan, run again (see Plan).
    const bool plan = n_tok == 1 && mtp1_.sched;
    const ReuseKey key{kv_span(pos0 + n_tok)};
    ggml_backend_sched* sc = plan ? mtp1_.sched : sched_;
    if (plan && !(mtp1_.g.ctx && mtp1_.key == key)) {
        drop_plan(mtp1_);
        mtp1_.g = build_mtp(1, pos0, &mtp1_.mem);
        mtp1_.g.kept = true;
        mtp1_.key = key;
        ggml_backend_sched_reset(sc);
        if (!ggml_backend_sched_alloc_graph(sc, mtp1_.g.gf)) { drop_plan(mtp1_); throw std::runtime_error("failed to allocate compute memory"); }
    }
    Graph built;
    if (!plan) {
        built = build_mtp(n_tok, pos0);
        drop_reuse();
        ggml_backend_sched_reset(sched_);
        if (!ggml_backend_sched_alloc_graph(sched_, built.gf)) { ggml_free(built.ctx); throw std::runtime_error("failed to allocate compute memory"); }
    }
    Graph& g = plan ? mtp1_.g : built;
    std::vector<int32_t> positions(n_tok);
    for (int i = 0; i < n_tok; ++i) positions[i] = pos0 + i;
    ggml_backend_tensor_set(g.inp, ids, 0, sizeof(int32_t) * n_tok);
    ggml_backend_tensor_set(g.pos, positions.data(), 0, sizeof(int32_t) * n_tok);
    ggml_backend_tensor_set(g.h_in, h, 0, sizeof(float) * (size_t)hp.hidden_width() * n_tok);
    if (g.mask) {
        const int n_kv = kv_span(pos0 + n_tok);                    // padded columns are future: -inf
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask((size_t)n_kv * n_tok);
        for (int t = 0; t < n_tok; ++t)
            for (int j = 0; j < n_kv; ++j) mask[(size_t)t * n_kv + j] = j <= pos0 + t ? zero : ninf;
        ggml_backend_tensor_set(g.mask, mask.data(), 0, sizeof(ggml_fp16_t) * mask.size());
    }
    set_n_threads(cpu_, n_threads);
    if (ggml_backend_sched_graph_compute(sc, g.gf) != GGML_STATUS_SUCCESS) {
        if (plan) drop_plan(mtp1_); else ggml_free(g.ctx);
        throw std::runtime_error("graph compute failed");
    }
    std::vector<float> out(hp.n_vocab);
    ggml_backend_tensor_get(g.logits, out.data(), 0, sizeof(float) * hp.n_vocab);
    if (h_out) {
        h_out->resize((size_t)hp.hidden_width() * n_tok);
        ggml_backend_tensor_get(g.hidden, h_out->data(), 0, sizeof(float) * h_out->size());
    }
    if (!plan) ggml_free(g.ctx);
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
// The KV cache (and the hybrid models' fixed-size running memories) at `ctx` positions, for n_seq conversations:
// each layer's part on the device that computes the layer (layer_dev_), one buffer per device. Cleared to zero.
void Model::make_kv(int ctx, int n_seq) {
    const Hparams& hp = hp_;
    const int nl_all = hp.n_layer + (mtp_ ? 1 : 0), G = (int)gpus_.size(), hd = hp.head_dim;
    const ggml_type kv_type = (ggml_type)kv_t_;
    ggml_init_params kp{ (size_t)3 * nl_all * n_seq * ggml_tensor_overhead() + 1024, nullptr, /*no_alloc=*/true };
    kvctx_ = ggml_init(kp);
    std::vector<std::vector<ggml_tensor*>> kv_t(G + 1);              // per GPU, then RAM
    for (int i = 0; i < nl_all; ++i) {
        auto& L = i < hp.n_layer ? layers_[i] : *mtp_;
        auto& dst = kv_t[layer_dev_[i] >= 0 ? layer_dev_[i] : G];
        if (L.ple_key) {                                              // Flash-Next: the PLE convolution's last inputs
            L.ple_states.clear();
            for (int s = 0; s < n_seq; ++s) {
                L.ple_states.push_back(ggml_new_tensor_2d(kvctx_, GGML_TYPE_F32, (int64_t)hp.hc * hp.n_embd, hp.ple_state_len()));
                dst.push_back(L.ple_states.back());
            }
        }
        if (L.recurrent) {                                            // DeltaNet: fixed-size state, always f32
            L.conv_states.clear(); L.ssm_states.clear();
            for (int s = 0; s < n_seq; ++s) {                         // one running memory per conversation
                L.conv_states.push_back(ggml_new_tensor_2d(kvctx_, GGML_TYPE_F32, hp.ssm_conv - 1, hp.ssm_conv_dim()));
                L.ssm_states.push_back(hp.mamba ? ggml_new_tensor_3d(kvctx_, GGML_TYPE_F32, hp.m_state, hp.m_inner / hp.m_heads, hp.m_heads)
                                                : ggml_new_tensor_3d(kvctx_, GGML_TYPE_F32, hp.ssm_state, hp.ssm_state, hp.ssm_v_heads));
                dst.push_back(L.conv_states.back());
                dst.push_back(L.ssm_states.back());
            }
            L.conv_state = L.conv_states[0];
            L.ssm_state_t = L.ssm_states[0];
            continue;
        }
        if (L.kind == Hparams::kFfn) continue;                       // Nemotron's expert blocks keep nothing
        L.k_cache = ggml_new_tensor_3d(kvctx_, kv_type, hd, hp.n_head_kv, ctx);
        dst.push_back(L.k_cache);
        if (hp.mla) { L.v_cache = nullptr; continue; }               // MLA: the values are part of the one cache
        L.v_cache = ggml_new_tensor_3d(kvctx_, kv_type, hd, hp.n_head_kv, ctx);
        dst.push_back(L.v_cache);
    }
    kv_bytes_ = 0;
    for (auto& g : place_.gpus) g.kv = 0;
    for (int tier = 0; tier <= G; ++tier) {
        auto& ts = kv_t[tier];
        if (ts.empty()) continue;
        ggml_backend_buffer* b = ggml_backend_buft_alloc_buffer_n(kv_bufts_[tier], ts.data(), (int)ts.size());
        if (!b) throw std::runtime_error("not enough memory for the KV cache (try a smaller context)");
        ggml_backend_buffer_clear(b, 0);
        kv_bytes_ += ggml_backend_buffer_get_size(b);
        if (tier < G) place_.gpus[tier].kv += ggml_backend_buffer_get_size(b);
        kvbufs_.push_back(b);
    }
}

size_t Model::kv_bytes_per_token(size_t* in_vram) const {
    size_t all = 0, vram = 0;
    const int nl_all = hp_.n_layer + (mtp_ ? 1 : 0);
    for (int i = 0; i < nl_all; ++i) {
        const auto& L = i < hp_.n_layer ? layers_[i] : *mtp_;
        if (L.recurrent || !L.k_cache) continue;
        const size_t b = L.k_cache->nb[2] + (L.v_cache ? L.v_cache->nb[2] : 0);
        all += b;
        if (layer_dev_[i] >= 0) vram += b;
    }
    if (in_vram) *in_vram = vram;
    return all;
}

void Model::free_kv() {
    for (auto* b : kvbufs_) if (b) ggml_backend_buffer_free(b);
    kvbufs_.clear();
    if (kvctx_) ggml_free(kvctx_);
    kvctx_ = nullptr;
}

// Live context length (see the header): what's in the cache goes to RAM, the cache is made again at the new size
// on the same devices, and it comes back. Positions stay where they are, so only the cells in use (up to hi_) are
// copied; the running memories of hybrid layers are copied whole. If the new size doesn't fit, the old one is made
// again and the conversations are as they were.
void Model::resize_ctx(int new_ctx) {
    std::lock_guard<std::recursive_mutex> dev_lock(device_mutex());
    if (new_ctx == max_ctx_) return;
    if (new_ctx < 256) throw std::runtime_error("the context must be at least 256 tokens");
    if (hi_ > new_ctx)
        throw std::runtime_error("the conversations in the cache use " + std::to_string(hi_) + " tokens, more than " +
                                 std::to_string(new_ctx) + ": start a new chat first, or choose a longer context");
    release_pending();
    drop_plans();                                               // (the kept graphs point at the cache that is about to be freed)
    const Hparams& hp = hp_;
    const int nl_all = hp.n_layer + (mtp_ ? 1 : 0), n_seq = n_seq_max();
    // 1. what to keep, in RAM
    struct Kept { std::vector<uint8_t> k, v; std::vector<std::vector<uint8_t>> conv, ssm, ple; };
    std::vector<Kept> kept(nl_all);
    auto get = [](ggml_tensor* t, size_t bytes) { std::vector<uint8_t> d(bytes); if (bytes) ggml_backend_tensor_get(t, d.data(), 0, bytes); return d; };
    for (int i = 0; i < nl_all; ++i) {
        auto& L = i < hp.n_layer ? layers_[i] : *mtp_;
        for (ggml_tensor* t : L.ple_states) kept[i].ple.push_back(get(t, ggml_nbytes(t)));
        if (L.recurrent) {
            for (int s = 0; s < n_seq; ++s) {
                kept[i].conv.push_back(get(L.conv_states[s], ggml_nbytes(L.conv_states[s])));
                kept[i].ssm.push_back(get(L.ssm_states[s], ggml_nbytes(L.ssm_states[s])));
            }
        } else if (L.k_cache) {
            kept[i].k = get(L.k_cache, (size_t)hi_ * L.k_cache->nb[2]);
            if (L.v_cache) kept[i].v = get(L.v_cache, (size_t)hi_ * L.v_cache->nb[2]);
        }
    }
    // 2. the cache again, at the new size (or the old one if that doesn't fit)
    const int old_ctx = max_ctx_;
    free_kv();
    std::string err;
    try { make_kv(new_ctx, n_seq); }
    catch (const std::exception& e) {
        err = e.what();
        free_kv();
        make_kv(old_ctx, n_seq);                                      // (it fitted a moment ago)
        new_ctx = old_ctx;
    }
    // 3. back in
    for (int i = 0; i < nl_all; ++i) {
        auto& L = i < hp.n_layer ? layers_[i] : *mtp_;
        for (size_t s = 0; s < L.ple_states.size() && s < kept[i].ple.size(); ++s)
            ggml_backend_tensor_set(L.ple_states[s], kept[i].ple[s].data(), 0, kept[i].ple[s].size());
        if (L.recurrent) {
            for (int s = 0; s < n_seq; ++s) {
                ggml_backend_tensor_set(L.conv_states[s], kept[i].conv[s].data(), 0, kept[i].conv[s].size());
                ggml_backend_tensor_set(L.ssm_states[s], kept[i].ssm[s].data(), 0, kept[i].ssm[s].size());
            }
        } else if (L.k_cache) {
            if (!kept[i].k.empty()) ggml_backend_tensor_set(L.k_cache, kept[i].k.data(), 0, kept[i].k.size());
            if (!kept[i].v.empty()) ggml_backend_tensor_set(L.v_cache, kept[i].v.data(), 0, kept[i].v.size());
        }
    }
    if (!err.empty()) throw std::runtime_error("the context stays at " + std::to_string(old_ctx) + " tokens: " + err);
    // 4. the cells: new ones free, or the unused end dropped
    cell_seq_.resize(new_ctx, -1);
    cell_pos_.resize(new_ctx, -1);
    n_free_ += new_ctx - max_ctx_;
    max_ctx_ = new_ctx;
    free_lo_ = std::min(free_lo_, max_ctx_);
    // 5. compute memory for the largest graph at the new size
    const int n_res = std::min(batch_, max_ctx_);
    Graph worst = build(n_res, max_ctx_, true);
    pin_cpu_layers(worst);
    drop_reuse();
    ggml_backend_sched_reset(sched_);
    const bool ok = ggml_backend_sched_reserve(sched_, worst.gf);
    ggml_free(worst.ctx);
    if (!ok) throw std::runtime_error("failed to reserve compute memory for the new context");
}

void Model::release_pending() {
    if (pending_) { if (!((Graph*)pending_)->kept) ggml_free(((Graph*)pending_)->ctx); delete (Graph*)pending_; pending_ = nullptr; }
}

void Model::accept(int n_keep) {
    if (!pending_) return;
    Graph& g = *(Graph*)pending_;
    const int d = g.n_tok - n_keep;                               // tokens to drop
    std::lock_guard<std::recursive_mutex> dev_lock(device_mutex());
    if (d > 0) {
        // (checked before anything is rewound, so a refusal leaves the state whole)
        for (auto& [li, rb] : g.rollback) if (d >= (int)rb.size()) throw std::runtime_error("speculative rollback deeper than its snapshots");
        for (auto& [li, rb] : g.ple_rollback) if (d >= (int)rb.size()) throw std::runtime_error("speculative rollback deeper than its snapshots");
        n_past_ -= d;                                             // attention layers: just forget their keys/values
        free_cells_from(0, n_past_);
        for (auto& [li, rb] : g.rollback) {                       // DeltaNet layers: restore the memory d tokens back
            if (d >= (int)rb.size()) throw std::runtime_error("speculative rollback deeper than its snapshots");
            Layer& L = layers_[li];
            ggml_backend_tensor_copy(rb[d].first, L.ssm_state_t);
            ggml_backend_tensor_copy(rb[d].second, L.conv_state);
        }
        for (auto& [li, rb] : g.ple_rollback) {                   // Flash-Next's PLE convolutions: their kept inputs too
            if (d >= (int)rb.size()) throw std::runtime_error("speculative rollback deeper than its snapshots");
            ggml_backend_tensor_copy(rb[d], layers_[li].ple_states[0]);
        }
    }
    release_pending();
}

Model::EvalOut Model::eval_ex(const int32_t* ids, int n_tok, int n_threads, int n_logits, bool want_hidden, int snapshots,
                              std::vector<int32_t>* routing) {
    std::lock_guard<std::recursive_mutex> dev_lock(device_mutex());   // another loaded model waits for this step
    release_pending();                                            // an unanswered check counts as fully accepted
    return step(ids, n_tok, n_threads, n_logits, want_hidden, snapshots, routing, nullptr);
}

std::vector<std::vector<float>> Model::eval_seqs(const std::vector<SeqInput>& in, int n_threads, std::vector<int32_t>* routing) {
    std::lock_guard<std::recursive_mutex> dev_lock(device_mutex());
    release_pending();
    std::vector<std::vector<float>> out(in.size());
    if (routing) routing->clear();
    size_t need = 0;
    for (const SeqInput& x : in) {
        if (x.seq < 0 || x.seq >= n_seq_max()) throw std::runtime_error("no such conversation: " + std::to_string(x.seq));
        need += x.ids.size();
    }
    if ((int)need > n_free_) throw std::runtime_error("context full (" + std::to_string(max_ctx_) + " tokens, shared by all conversations)");
    // Packed into batches of at most batch_ tokens; a long input is spread over several, and its logits come
    // from the batch holding its last token.
    std::vector<size_t> done(in.size(), 0);
    std::vector<int32_t> part;
    for (;;) {
        std::vector<int32_t> ids;
        std::vector<SeqGroup> groups;
        std::vector<int> which;
        for (size_t i = 0; i < in.size() && (int)ids.size() < batch_; ++i) {
            const size_t left = in[i].ids.size() - done[i];
            if (!left) continue;
            const int n = (int)std::min<size_t>(left, (size_t)(batch_ - (int)ids.size()));
            ids.insert(ids.end(), in[i].ids.begin() + done[i], in[i].ids.begin() + done[i] + n);
            groups.push_back({in[i].seq, n});
            which.push_back((int)i);
            done[i] += n;
        }
        if (ids.empty()) break;
        EvalOut r = step(ids.data(), (int)ids.size(), n_threads, 1, false, 1, routing ? &part : nullptr, &groups);
        if (routing) routing->insert(routing->end(), part.begin(), part.end());
        const size_t V = (size_t)hp_.n_vocab;
        for (size_t k = 0; k < which.size(); ++k)
            if (done[which[k]] == in[which[k]].ids.size())
                out[which[k]].assign(r.logits.begin() + k * V, r.logits.begin() + (k + 1) * V);
    }
    return out;
}

void Model::drop_reuse() {
    if (reuse_.ctx) ggml_free(reuse_.ctx);
    reuse_ = Graph();
}

void Model::drop_plan(Plan& p) {
    if (!p.g.ctx) return;
    if (pending_ && ((Graph*)pending_)->ctx == p.g.ctx) { delete (Graph*)pending_; pending_ = nullptr; }   // (it was this graph)
    ggml_free(p.g.ctx);
    p.g = Graph();
}

void Model::drop_plans() { drop_reuse(); drop_plan(check_); drop_plan(mtp1_); }

Model::EvalOut Model::step(const int32_t* ids, int n_tok, int n_threads, int n_logits, bool want_hidden, int snapshots,
                           std::vector<int32_t>* routing, const std::vector<SeqGroup>* groups) {
    const Hparams& hp = hp_;
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    auto t0 = clk::now();
    // Each token's conversation and position, and a cache cell for it.
    std::vector<int> tok_seq, tok_pos, cells;
    tok_seq.reserve(n_tok); tok_pos.reserve(n_tok); cells.reserve(n_tok);
    std::vector<std::pair<int, int>> undo;                        // (conversation, tokens before) if this step fails
    struct Undo {                                                 // on an exception: give the cells back
        Model& m; std::vector<std::pair<int, int>>& u; bool on = true;
        ~Undo() { if (on) for (auto [s, n] : u) m.free_cells_from(s, n); }
    } undo_guard{*this, undo};
    {
        const SeqGroup one{0, n_tok};
        const std::vector<SeqGroup> single(1, one);
        for (const SeqGroup& gr : groups ? *groups : single) {
            undo.push_back({gr.seq, seq_past(gr.seq)});
            const int p0 = seq_past(gr.seq);
            std::vector<int> got;
            try { got = alloc_cells(gr.seq, gr.n); }
            catch (...) { for (auto [s, n] : undo) free_cells_from(s, n); throw; }
            for (int i = 0; i < gr.n; ++i) { tok_seq.push_back(gr.seq); tok_pos.push_back(p0 + i); cells.push_back(got[i]); }
        }
    }
    // Each token's rotary position. Text: the conversation's next position. An image's word: (start, start + row,
    // start + column), start being where the image began; after the image, text continues at
    // start + max(rows, cols). Without images that is the token's index, as before.
    std::vector<std::array<int, 3>> rope(n_tok);
    bool any_image = false;
    {
        std::map<int, int> next;                                  // per conversation in this step
        for (int t = 0; t < n_tok; ++t) {
            const int s = tok_seq[t];
            auto it = next.find(s);
            if (it == next.end()) it = next.emplace(s, seq_next_[s]).first;
            const int32_t id = ids[t];
            if (!is_image_id(id)) { rope[t] = {it->second, it->second, it->second}; it->second++; continue; }
            const ImageRec* im = image_of(id);
            if (!im) throw std::runtime_error("an image this conversation refers to is no longer known; send it again");
            if (!hp.mrope) throw std::runtime_error("this model doesn't read images");
            any_image = true;
            const int k = id - im->first, row = k / im->cols, col = k % im->cols;
            // the image's start: this word's own position if it is the first, else the first word's
            int start;
            if (k == 0) start = it->second;
            else {
                const int back = t - k;                           // the first word, in this step or earlier
                if (back >= 0 && tok_seq[back] == s) start = rope[back][0];
                else {
                    const int idx = tok_pos[t] - k;
                    if (idx < 0 || idx >= (int)seq_rope_[s].size()) throw std::runtime_error("an image's words arrived out of order");
                    start = seq_rope_[s][idx][0];
                }
            }
            rope[t] = {start, start + row, start + col};
            it->second = start + std::max(im->rows, im->cols);
        }
    }
    const int n_kv_end = cells_end(), n_kv = kv_span(n_kv_end);
    std::vector<int32_t> live_routing;                            // live hot experts need the picks even if the caller doesn't
    if (!routing && !live_.empty()) routing = &live_routing;
    build_images_ = any_image; build_mrope_ = any_image;
    if (!qsa_.empty()) {                                          // QSA: who's who in this step, and is anyone past the budget
        qsa_batch_ = QsaBatch();
        qsa_long_ = false;
        int rmin = 1 << 30;
        for (const QsaLayer& Q : qsa_) rmin = std::min(rmin, Q.ratio);
        for (int t = 0; t < n_tok; ++t) {
            qsa_batch_.seq.push_back(tok_seq[t]); qsa_batch_.pos.push_back(tok_pos[t]); qsa_batch_.rope.push_back(rope[t]);
            qsa_batch_.at[{tok_seq[t], tok_pos[t]}] = t;
            if ((tok_pos[t] + 1) / rmin > hp.qsa_budget / rmin) qsa_long_ = true;
        }
    }
    // Replies, one token at a time, make the same graph over and over: positions, cache cells and masks are inputs,
    // and the cache span grows in steps of 256. So it is built, placed and allocated once and then run again, until
    // something about it changes (any other graph, or a new span). Saves ~3 ms a token. TIERLM_REUSE=0: always build.
    static const bool reuse_on = !(std::getenv("TIERLM_REUSE") && !std::strcmp(std::getenv("TIERLM_REUSE"), "0")) &&
                                 !std::getenv("TIERLM_DUMP_SCHED");
    const bool cacheable = reuse_on && n_tok == 1 && !groups && !any_image && snapshots <= 1;
    // A check of guesses (an undoable step) has a scheduler of its own, where its graph stays placed (see Plan).
    const bool plan = reuse_on && check_.sched && snapshots > 1 && !groups && !any_image;
    const ReuseKey key{n_kv, routing != nullptr, n_logits, qsa_long_, want_hidden, n_tok, snapshots};
    const bool reuse = plan ? check_.g.ctx && key == check_.key : cacheable && reuse_.ctx && key == reuse_key_;
    ggml_backend_sched* const sc = plan ? check_.sched : sched_;
    Graph built;
    if (!reuse) {
        if (plan) drop_plan(check_);
        built = build(n_tok, n_kv_end, routing != nullptr, n_logits, want_hidden, snapshots, groups, plan ? &check_.mem : nullptr);
        if (plan) { built.kept = true; check_.g = built; check_.key = key; }
    }
    Graph& g = plan ? check_.g : reuse ? reuse_ : built;
    build_images_ = build_mrope_ = false;
    g.n_tok = n_tok;
    auto t1 = clk::now();

    if (!reuse) {
        ggml_backend_sched_reset(sc);
        pin_cpu_layers(g, sc);
        if (!ggml_backend_sched_alloc_graph(sc, g.gf)) {
            if (plan) drop_plan(check_); else ggml_free(g.ctx);
            throw std::runtime_error("failed to allocate compute memory");
        }
    }
    auto t2 = clk::now();
    if (std::getenv("TIERLM_DUMP_SCHED") && n_tok == 1) {           // debugging: where the work switches device
        ggml_backend_t prev = nullptr;
        int layer = -1, shown = 0;
        for (int i = 0; i < ggml_graph_n_nodes(g.gf) && shown < 60; ++i) {
            ggml_tensor* t = ggml_graph_node(g.gf, i);
            if (t->op == GGML_OP_FLASH_ATTN_EXT || (t->op == GGML_OP_SOFT_MAX && !flash_attn_)) ++layer;
            ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sc, t);
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
        std::fprintf(stderr, "splits %d\n", ggml_backend_sched_get_n_splits(sc));
        // TIERLM_DUMP_SCHED=2: also every step of layers 1-4 (op, where it runs, shape), and the step count per device
        if (std::atoi(std::getenv("TIERLM_DUMP_SCHED")) >= 2) {
            std::map<std::string, int> per_dev;
            layer = -1;
            for (int i = 0; i < ggml_graph_n_nodes(g.gf); ++i) {
                ggml_tensor* t = ggml_graph_node(g.gf, i);
                if (t->op == GGML_OP_FLASH_ATTN_EXT || (t->op == GGML_OP_SOFT_MAX && !flash_attn_)) ++layer;
                ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sc, t);
                const bool view = t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE;
                const std::string dev = b ? ggml_backend_name(b) : "?";
                if (!view) per_dev[dev]++;
                if (i < 400)
                    std::fprintf(stderr, "node %4d %-8s %-16s %-28s %lld x %lld x %lld%s\n", i, dev.c_str(), ggml_op_desc(t), t->name,
                                 (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], view ? " (view)" : "");
            }
            for (const auto& [d, n] : per_dev) std::fprintf(stderr, "steps on %s: %d per token\n", d.c_str(), n);
        }
    }

    if (g.img) {                                                  // images: their words replace the embeddings
        std::vector<int32_t> tid(ids, ids + n_tok);
        std::vector<float> keep(n_tok, 1.0f), words((size_t)hp.n_embd * n_tok, 0.0f);
        for (int t = 0; t < n_tok; ++t) {
            if (!is_image_id(ids[t])) continue;
            const ImageRec* im = image_of(ids[t]);
            std::copy_n(im->words.data() + (size_t)(ids[t] - im->first) * hp.n_embd, hp.n_embd, words.data() + (size_t)t * hp.n_embd);
            tid[t] = 0; keep[t] = 0.0f;
        }
        ggml_backend_tensor_set(g.inp, tid.data(), 0, sizeof(int32_t) * n_tok);
        ggml_backend_tensor_set(g.keep, keep.data(), 0, sizeof(float) * n_tok);
        ggml_backend_tensor_set(g.img, words.data(), 0, sizeof(float) * words.size());
    } else ggml_backend_tensor_set(g.inp, ids, 0, sizeof(int32_t) * n_tok);
    {
        std::vector<int32_t> pos(g.mrope ? 4 * (size_t)n_tok : n_tok, 0);
        for (int t = 0; t < n_tok; ++t) {
            pos[t] = rope[t][0];
            if (g.mrope) { pos[n_tok + t] = rope[t][1]; pos[2 * n_tok + t] = rope[t][2]; }
        }
        // (Nemotron rotates no positions: nothing reads them, so the graph gave them no memory)
        if (g.pos->buffer) ggml_backend_tensor_set(g.pos, pos.data(), 0, sizeof(int32_t) * pos.size());
    }
    if (g.cells->buffer) ggml_backend_tensor_set(g.cells, cells.data(), 0, sizeof(int32_t) * n_tok);
    if (g.zero_id && g.zero_id->buffer) { const int32_t z = 0; ggml_backend_tensor_set(g.zero_id, &z, 0, sizeof z); }
    if (g.ngram) {                                                 // Flash-Next: each token's n-gram rows
        const int NH = hp.ngram.heads();
        std::vector<int32_t> rows((size_t)NH * n_tok);
        std::map<int, std::vector<int32_t>> hist;                 // each conversation's tokens, this step's included
        for (int t = 0; t < n_tok; ++t) {
            const int s = tok_seq[t];
            auto it = hist.find(s);
            if (it == hist.end()) it = hist.emplace(s, std::vector<int32_t>(seq_tok_[s].begin(), seq_tok_[s].begin() + std::min((size_t)tok_pos[t], seq_tok_[s].size()))).first;
            auto& h = it->second;
            if ((int)h.size() != tok_pos[t]) throw std::runtime_error("the n-gram history is out of step with the conversation");
            h.push_back(is_image_id(ids[t]) ? (hp.ple_image_tok >= 0 ? hp.ple_image_tok : hp.ngram.eos) : ids[t]);
            hp.ngram.rows(h.data(), (int64_t)h.size() - 1, rows.data() + (size_t)t * NH);
        }
        ggml_backend_tensor_set(g.ngram, rows.data(), 0, sizeof(int32_t) * rows.size());
        // The table on the SSD tier: ask the drive for every row this step reads, all at once, before the graph
        // gathers them one page fault after another (16 random reads a token; Flash-Next IQ3_XXS on a 5090 ran
        // 40% slower until the rows it uses were in the file cache).
        if (place_.ssd_ngram && ple_embd_->buffer && ggml_backend_buffer_is_host(ple_embd_->buffer) && !std::getenv("TIERLM_NO_NGRAM_PREFETCH")) {
            std::vector<std::pair<const void*, size_t>> r;
            r.reserve(rows.size());
            for (int32_t id : rows) r.push_back({(const char*)ple_embd_->data + (size_t)id * ple_embd_->nb[1], ple_embd_->nb[1]});
            MappedFile::prefetch_many(r);
        }
    }
    fill_masks(g, n_kv, tok_seq, tok_pos);
    if (g.out_idx) {                                               // each conversation's last token
        std::vector<int32_t> idx;
        int t = 0;
        for (const SeqGroup& gr : *groups) { t += gr.n; idx.push_back(t - 1); }
        ggml_backend_tensor_set(g.out_idx, idx.data(), 0, sizeof(int32_t) * idx.size());
    }
    if (g.rope_ff) ggml_backend_tensor_set(g.rope_ff, hp.rope_ff.data(), 0, sizeof(float) * hp.rope_ff.size());
    set_n_threads(cpu_, n_threads);
    n_threads_last_ = n_threads;
    auto t3 = clk::now();

    // TIERLM_TRACE=1: each named result's size, sum, RMS and the last token's first numbers, as it's computed
    static const bool trace = std::getenv("TIERLM_TRACE") != nullptr;
    if (trace) {
        ggml_backend_sched_set_eval_callback(sc, [](ggml_tensor* t, bool ask, void*) -> bool {
            const char* n = ggml_get_name(t);
            const bool mine = std::strncmp(n, "tr:", 3) == 0 && !std::strchr(n, ' ');   // (not the views made from it)
            if (ask) return mine;
            if (!mine || t->type != GGML_TYPE_F32) return true;
            const int64_t ne = ggml_nelements(t);
            std::vector<float> v((size_t)ne);
            if (ggml_is_contiguous(t)) ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
            else return true;
            double s = 0, ss = 0;
            for (float f : v) { s += f; ss += (double)f * f; }
            const int64_t last = ne - t->ne[0];                    // the last column's first numbers
            std::fprintf(stderr, "trace %-16s [%lld,%lld,%lld] sum %.6g rms %.6g | last: %.5g %.5g %.5g %.5g\n", n + 3,
                         (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], s, std::sqrt(ss / std::max<int64_t>(1, ne)),
                         v[(size_t)last], v[(size_t)last + 1], v[(size_t)last + 2], v[(size_t)last + 3]);
            return true;
        }, nullptr);
    }
    // TIERLM_OPPROF=1 (developers): time per kind of step (op and device), each one waited for, so the totals are
    // slower than a real run but show what grows with the number of tokens in a step. Printed every 32 steps of a size.
    static const bool opprof = std::getenv("TIERLM_OPPROF") != nullptr;
    struct OpProf { Model* m; ggml_backend_sched* sc; clk::time_point last; std::map<std::string, double>* acc; };
    static std::map<int, std::pair<int, std::map<std::string, double>>> opstats;
    OpProf op_state{this, sc, clk::now(), &opstats[n_tok].second};
    if (opprof && !trace) {
        ggml_backend_sched_set_eval_callback(sc, [](ggml_tensor* t, bool ask, void* ud) -> bool {
            const bool view = t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE ||
                              t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE;
            if (ask) return !view;
            auto* st = (OpProf*)ud;
            const auto now = clk::now();
            ggml_backend_t b = ggml_backend_sched_get_tensor_backend(st->sc, t);
            std::string key = std::string(b ? ggml_backend_name(b) : "?") + " " + ggml_op_desc(t);
            if (t->op == GGML_OP_MUL_MAT_ID || t->op == GGML_OP_MUL_MAT) key += std::string(" ") + ggml_type_name(t->src[0]->type);
            (*st->acc)[key] += std::chrono::duration<double, std::milli>(now - st->last).count();
            st->last = now;
            return true;
        }, &op_state);
    }
    if (ggml_backend_sched_graph_compute(sc, g.gf) != GGML_STATUS_SUCCESS) {
        if (plan) drop_plan(check_); else if (reuse) drop_reuse(); else ggml_free(g.ctx);
        throw std::runtime_error("graph compute failed");
    }
    if (opprof && !trace) {
        ggml_backend_sched_set_eval_callback(sc, nullptr, nullptr);
        auto& [cnt, acc] = opstats[n_tok];
        if (++cnt == (n_tok > 16 ? 3 : 32)) {                      // (prompt batches are few: every 3)
            std::vector<std::pair<double, std::string>> v;
            double tot = 0;
            for (auto& [k, ms] : acc) { v.push_back({ms / cnt, k}); tot += ms / cnt; }
            std::sort(v.rbegin(), v.rend());
            std::fprintf(stderr, "opprof: %d-token steps, %.2f ms each (each step waited for):", n_tok, tot);
            for (size_t i = 0; i < v.size() && i < 14; ++i) std::fprintf(stderr, "%s %s %.2f", i ? "," : "", v[i].second.c_str(), v[i].first);
            std::fprintf(stderr, "\n");
            cnt = 0; acc.clear();
        }
    }
    auto t4 = clk::now();
    EvalOut res;
    res.logits.resize((size_t)hp.n_vocab * g.logits->ne[1]);
    ggml_backend_tensor_get(g.logits, res.logits.data(), 0, sizeof(float) * res.logits.size());
    if (g.hidden) {
        res.hidden.resize((size_t)hp.hidden_width() * n_tok);
        ggml_backend_tensor_get(g.hidden, res.hidden.data(), 0, sizeof(float) * res.hidden.size());
    }
    if (routing) {
        // [token][layer][pick] for every layer; layers without experts (Flint's dense first layers) get -1
        const int L = hp.is_moe() ? hp.n_layer : 0, K = hp.n_expert_used, R = (int)g.orders.size();
        routing->assign((size_t)n_tok * L * K, -1);
        if (g.routing) {                                                  // laid out [moe layer][token][pick]
            std::vector<int32_t> buf((size_t)ggml_nelements(g.routing));
            ggml_backend_tensor_get(g.routing, buf.data(), 0, sizeof(int32_t) * buf.size());
            size_t at = 0;
            for (int r = 0; r < R; ++r) {
                const int l = g.order_layers[r];
                const int kl = std::min(K, hp.n_used_at(l));              // (a block that picks fewer leaves the rest -1)
                for (int t = 0; t < n_tok; ++t)
                    for (int k = 0; k < kl; ++k)
                        (*routing)[((size_t)t * L + l) * K + k] = buf[at + (size_t)t * kl + k];
                at += (size_t)kl * n_tok;
            }
        }
    }
    if (routing && !live_.empty()) adapt_hot(*routing, n_tok);
    for (auto [qi, kr] : g.qsa_k) {                                // QSA: this step's indexer keys, kept by position
        const int D = hp.qsa_dim;
        std::vector<float> k((size_t)D * n_tok);
        ggml_backend_tensor_get(kr, k.data(), 0, k.size() * sizeof(float));
        for (int t = 0; t < n_tok; ++t) {
            auto& v = qsa_[qi].keys[tok_seq[t]];
            v.resize((size_t)tok_pos[t] * D);
            auto& bl = qsa_[qi].blocks[tok_seq[t]];                // blocks from here on are made again
            if (bl.size() > (size_t)(tok_pos[t] / qsa_[qi].ratio) * D) bl.resize((size_t)(tok_pos[t] / qsa_[qi].ratio) * D);
            v.insert(v.end(), k.begin() + (size_t)t * D, k.begin() + (size_t)(t + 1) * D);
        }
    }
    if (snapshots > 1) { pending_ = new Graph(g); }              // kept for accept()
    else if (plan) {}                                             // (a check that can't be undone: the plan stays)
    else if (cacheable) { if (!reuse) { reuse_ = g; reuse_key_ = key; } }   // kept to run again
    else ggml_free(g.ctx);
    auto t5 = clk::now();
    stats_.calls++; stats_.tokens += n_tok;
    stats_.build_ms += ms(t0, t1); stats_.alloc_ms += ms(t1, t2); stats_.input_ms += ms(t2, t3);
    stats_.compute_ms += ms(t3, t4); stats_.output_ms += ms(t4, t5);
    stats_.splits = ggml_backend_sched_get_n_splits(sc); stats_.copies = ggml_backend_sched_get_n_copies(sc);
    undo_guard.on = false;
    if (hp.any_ple())                                             // Flash-Next: the tokens, for later n-grams
        for (int t = 0; t < n_tok; ++t) {
            auto& v = seq_tok_[tok_seq[t]];
            if ((int)v.size() != tok_pos[t]) v.resize(tok_pos[t]);
            v.push_back(is_image_id(ids[t]) ? (hp.ple_image_tok >= 0 ? hp.ple_image_tok : hp.ngram.eos) : ids[t]);
        }
    for (int t = 0; t < n_tok; ++t) {                            // the positions are now part of the conversations
        auto& r = seq_rope_[tok_seq[t]];
        if ((int)r.size() != tok_pos[t]) r.resize(tok_pos[t]);
        r.push_back(rope[t]);
        const auto& p = rope[t];
        seq_next_[tok_seq[t]] = std::max(seq_next_[tok_seq[t]], is_image_id(ids[t]) ? seq_next_[tok_seq[t]] : p[0] + 1);
    }
    for (int t = 0; t < n_tok; ++t)
        if (is_image_id(ids[t])) { const ImageRec* im = image_of(ids[t]); seq_next_[tok_seq[t]] = std::max(seq_next_[tok_seq[t]], rope[t][0] + std::max(im->rows, im->cols)); }
    for (QsaLayer& Q : qsa_)                                       // QSA: keys of the blocks finished so far, made once
                                                                   // (after the positions above, which they need)
        for (size_t s = 0; s < Q.keys.size(); ++s) {
            const int D = hp.qsa_dim, nb = (int)(Q.keys[s].size() / D) / Q.ratio;
            auto& bl = Q.blocks[s];
            std::vector<const float*> parts(Q.ratio);
            for (int bi = (int)(bl.size() / D); bi < nb; ++bi) {
                for (int j = 0; j < Q.ratio; ++j) parts[j] = Q.keys[s].data() + (size_t)(bi * Q.ratio + j) * D;
                bl.resize((size_t)(bi + 1) * D);
                qsa_block_key(Q, parts.data(), seq_rope_[s][bi * Q.ratio], bl.data() + (size_t)bi * D);
            }
        }
    n_past_ = seq_past(0);
    if (std::getenv("TIERLM_PROFILE")) profile_step(n_tok, ms(t0, t3), ms(t3, t4), ms(t4, t5), ms(t0, t1), ms(t1, t2));
    return res;
}


// Live hot experts: after each step, count which experts the router picked in each split layer, and move the most
// wanted ones that sit in RAM into VRAM in place of the least wanted ones there. Scores fade by 3% a token, so a
// conversation's current topic counts most; a newcomer must beat the expert it replaces clearly (x1.3 + 0.5),
// so two experts of about equal use don't keep swapping. Each step copies at most TIERLM_LIVE_MB (default 32 MB)
// and two experts per layer, starting at a different layer each step so all get their turn. The copy is from the
// RAM part (which holds every expert) into the VRAM slot; then the three small maps change. Results don't change:
// each pick is computed in whichever part holds it, as with a fixed set. Called with the device lock held, after
// the step's work is done, so no graph is using the slot.
void Model::adapt_hot(const std::vector<int32_t>& routing, int n_tok) {
    const Hparams& hp = hp_;
    const int E = hp.n_expert, K = hp.n_expert_used, NL = hp.n_layer;
    // How much to move per step: a share of the step's own time (TIERLM_LIVE_SHARE, default 8%), from how long the
    // last steps took and how fast moves went, at most TIERLM_LIVE_MB (default 32). Measured on an RTX 3070 laptop with
    // Qwen3.6-35B-A3B over five subjects: a fixed 32 MB per word spent a quarter of the time moving experts and ran
    // slower than not moving them at all (28.1 against 30.7 tok/s); 8 MB, about 8% of the time, was the fastest (33.1),
    // 4 MB as fast (fewer moves, a lower VRAM share), 16 MB a little slower.
    static const size_t cap_bytes = [] { const char* e = std::getenv("TIERLM_LIVE_MB"); return (size_t)((e ? std::atof(e) : 32.0) * 1024 * 1024); }();
    static const double share = [] { const char* e = std::getenv("TIERLM_LIVE_SHARE"); return e ? std::atof(e) : 0.08; }();
    static const bool eager = std::getenv("TIERLM_LIVE_EAGER") != nullptr;   // tests: swap at every chance
    const auto now0 = std::chrono::steady_clock::now();
    const bool prompt = n_tok > 16;                                             // (a speculative check is a few tokens: a word's step)
    if (!prompt && live_last_.time_since_epoch().count() != 0) {
        const double step = std::chrono::duration<double, std::milli>(now0 - live_last_).count();
        if (step > 0 && step < 5000) live_step_ms_ = live_step_ms_ > 0 ? 0.8 * live_step_ms_ + 0.2 * step : step;
    }
    // (what a step may spend adds up while unused, so a budget smaller than one expert still moves one now and then)
    size_t allowance = (size_t)8 << 20;                                         // (until moves have been timed)
    if (live_step_ms_ > 0 && live_ms_per_mb_ > 0) allowance = (size_t)(share * live_step_ms_ / live_ms_per_mb_ * 1024 * 1024);
    size_t budget_bytes = std::min(cap_bytes, live_credit_ + allowance);
    if (prompt || eager) budget_bytes = cap_bytes;                              // a prompt: a long step (tests: every chance)
    static const double prompt_ms = [] { const char* e = std::getenv("TIERLM_LIVE_PROMPT_MS"); return e ? std::atof(e) : 250.0; }();
    static const double prompt_hit = [] { const char* e = std::getenv("TIERLM_LIVE_PROMPT_HIT"); return e ? std::atof(e) : 0.62; }();
    const uint64_t picks0 = live_stats_.picks, hits0 = live_stats_.hits;
    static const int per_layer = [] { const char* e = std::getenv("TIERLM_LIVE_PER_LAYER"); return e ? std::max(0, std::atoi(e)) : 2; }();
    static const float margin = [] { const char* e = std::getenv("TIERLM_LIVE_MARGIN"); return e ? (float)std::atof(e) : 1.3f; }();
    if ((int)routing.size() < n_tok * NL * K) return;
    // The RAM parts are kept in the file's layout in plain host memory (see the repack choice in load), so an expert
    // is one contiguous slice to copy. Anything else is never read back (a repacked buffer can't be).
    for (const LiveLayer& ll : live_)
        for (const LivePart& p : ll.parts)
            if (!ggml_backend_buffer_is_host(p.cold->buffer)) return;
    const float decay = std::pow(0.97f, (float)n_tok);
    for (LiveLayer& ll : live_) {
        for (float& s : ll.score) s *= decay;
        for (int t = 0; t < n_tok; ++t)
            for (int k = 0; k < K; ++k) {
                const int e = routing[((size_t)t * NL + ll.layer) * K + k];
                if (e < 0 || e >= E) continue;
                ll.score[e] += 1.0f;
                live_stats_.picks++;
                if (ll.slot_of[e] >= 0) live_stats_.hits++;
            }
    }
    // After a prompt batch whose picks the VRAM experts served poorly (a new subject): as many moves as pay, so the
    // reply starts with the prompt's experts in VRAM instead of catching up 8% of a step at a time. How long it may
    // take goes by how poorly: nothing at 62% served or more, up to a quarter of a second at 45% or less (measured on
    // Flash-Next, RTX 5090: batches on the subject the experts suit were served 65-82%, a new subject's 42-54%; a
    // burst moves about 750 experts in 250 ms, and the next batch is then served 74-80%, so a long prompt pays once).
    // Not for a prompt the experts already suit: there the moves bring in the prompt's own experts in place of ones
    // the reply uses, and replies ran 4-6% slower.
    // TIERLM_LIVE_PROMPT_MS: the longest (0: never, as before 0.8.7); TIERLM_LIVE_PROMPT_HIT: the share where it starts.
    const uint64_t picks = live_stats_.picks - picks0, hits = live_stats_.hits - hits0;
    const double served = picks ? (double)hits / (double)picks : 1.0;
    const double burst_ms = prompt && !eager ? prompt_ms * std::min(1.0, std::max(0.0, (prompt_hit - served) / 0.17)) : 0.0;
    const bool burst = burst_ms >= 10.0;
    if (burst) budget_bytes = live_ms_per_mb_ > 0 ? (size_t)(burst_ms / live_ms_per_mb_ * 1024 * 1024) : (size_t)(burst_ms / 250.0 * 256) << 20;
    const auto t0 = std::chrono::steady_clock::now();
    size_t spent = 0;
    const int n = (int)live_.size();
    // The copies go on the GPU's queue without waiting for each (one wait at the end): the incoming expert's bytes are
    // staged first, since its RAM slot receives the outgoing one. (TIERLM_LIVE_SYNC=1: wait for each copy, as before.)
    static const bool sync_each = std::getenv("TIERLM_LIVE_SYNC") != nullptr;
    ggml_backend* gpu = sync_each || gpus_.empty() ? nullptr : gpus_[0];
    swap_buf_.clear();
    // (a burst is staged 64 MB at a time: the copies queued so far are waited for when the staging memory is full)
    const size_t stage = std::min(budget_bytes, (size_t)64 << 20) + ((size_t)64 << 20);
    if (gpu && swap_buf_.capacity() < stage) swap_buf_.reserve(stage);
    static int start = 0;
    start = n ? (start + 1) % n : 0;
    bool any = false;
    for (int pass = 0, moved = 1; moved && pass < (burst ? 64 : 1); ++pass) {     // (a burst: round after round while any layer moves one)
    moved = 0;
    for (int j = 0; j < n && spent < budget_bytes; ++j) {
        LiveLayer& ll = live_[(start + j) % n];
        size_t per_expert = 0;
        for (const LivePart& p : ll.parts) per_expert += 2 * p.hot->nb[2];   // (both ways)
        for (int s = 0; s < per_layer && spent + per_expert <= budget_bytes; ++s) {
            int in = -1, out = -1;                               // the best expert in RAM, the worst in VRAM
            for (int e = 0; e < E; ++e) {
                if (ll.slot_of[e] < 0) { if (in < 0 || ll.score[e] > ll.score[in]) in = e; }
                else if (out < 0 || ll.score[e] < ll.score[out]) out = e;
            }
            if (in < 0 || out < 0 || ll.score[in] <= (eager ? ll.score[out] : ll.score[out] * margin + 0.5f)) break;
            // Trade places: `in` from its RAM slot into out's VRAM slot, `out` back into that RAM slot.
            const int slot = ll.slot_of[out], cs = ll.cold_of[in];
            for (const LivePart& p : ll.parts) {
                const size_t per = p.hot->nb[2];
                char* ram = (char*)p.cold->data + (size_t)cs * per;
                if (gpu && p.hot->buffer && ggml_backend_buffer_get_type(p.hot->buffer) == ggml_backend_get_default_buffer_type(gpu)) {
                    if (swap_buf_.size() + per > swap_buf_.capacity()) { ggml_backend_synchronize(gpu); swap_buf_.clear(); }
                    const size_t off = swap_buf_.size();
                    swap_buf_.insert(swap_buf_.end(), ram, ram + per);          // (capacity reserved: never moves)
                    ggml_backend_tensor_get_async(gpu, p.hot, ram, (size_t)slot * per, per);
                    ggml_backend_tensor_set_async(gpu, p.hot, swap_buf_.data() + off, (size_t)slot * per, per);
                } else {
                    swap_buf_.assign(ram, ram + per);
                    ggml_backend_tensor_get(p.hot, ram, (size_t)slot * per, per);
                    ggml_backend_tensor_set(p.hot, swap_buf_.data(), (size_t)slot * per, per);
                }
            }
            const int32_t h_in = slot, h_out = ll.H, c_in = E - ll.H, c_out = cs;
            const float m_in = 1.0f, m_out = 0.0f;
            ggml_backend_tensor_set(ll.map_hot, &h_in, sizeof(int32_t) * in, sizeof(int32_t));
            ggml_backend_tensor_set(ll.map_hot, &h_out, sizeof(int32_t) * out, sizeof(int32_t));
            ggml_backend_tensor_set(ll.map_cold, &c_in, sizeof(int32_t) * in, sizeof(int32_t));
            ggml_backend_tensor_set(ll.map_cold, &c_out, sizeof(int32_t) * out, sizeof(int32_t));
            ggml_backend_tensor_set(ll.mask, &m_in, sizeof(float) * in, sizeof(float));
            ggml_backend_tensor_set(ll.mask, &m_out, sizeof(float) * out, sizeof(float));
            ll.slot_of[in] = slot; ll.slot_of[out] = -1; ll.in_slot[slot] = in;
            ll.cold_of[out] = cs; ll.cold_of[in] = -1;
            spent += per_expert;
            live_stats_.swaps++;
            any = true; ++moved;
            if (ll.layer < (int)place_.hot.size()) {             // the dashboard's map of VRAM experts
                auto& v = place_.hot[ll.layer];
                v = ll.in_slot;
                std::sort(v.begin(), v.end());
            }
        }
    }
    }
    if (any && gpu) ggml_backend_synchronize(gpu);                // the outgoing experts are in RAM before the CPU's next step
    const auto t1 = std::chrono::steady_clock::now();
    if (any) {
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        live_stats_.swap_ms += ms;
        const double mb = spent / (1024.0 * 1024.0);
        if (mb > 0) live_ms_per_mb_ = live_ms_per_mb_ > 0 ? 0.8 * live_ms_per_mb_ + 0.2 * (ms / mb) : ms / mb;
    }
    if (prompt && std::getenv("TIERLM_LIVE_TRACE"))                // (developers: what each prompt batch found and did)
        std::fprintf(stderr, "live: prompt batch of %d tokens, VRAM experts served %.0f%% of its picks; moved %zu MB in %.0f ms%s\n", n_tok,
                     picks ? 100.0 * hits / picks : 0.0, spent >> 20, std::chrono::duration<double, std::milli>(t1 - t0).count(), burst ? " (burst)" : "");
    live_last_ = t1;
    live_credit_ = prompt || eager ? 0 : budget_bytes - std::min(spent, budget_bytes);
}

}
