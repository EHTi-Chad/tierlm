#include "vision.h"
#include "backends.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>

namespace tierlm {

namespace {

struct Keys {
    gguf_context* g;
    std::string str(const std::string& k) const {
        int64_t id = gguf_find_key(g, k.c_str());
        return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING ? gguf_get_val_str(g, id) : "";
    }
    double num(const std::string& k, double def) const {
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
            case GGUF_TYPE_BOOL: return gguf_get_val_bool(g, id) ? 1 : 0;
            default: return def;
        }
    }
    bool floats(const std::string& k, float* out, int n) const {
        int64_t id = gguf_find_key(g, k.c_str());
        if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, id) != GGUF_TYPE_FLOAT32 ||
            (int)gguf_get_arr_n(g, id) != n) return false;
        std::memcpy(out, gguf_get_arr_data(g, id), sizeof(float) * n);
        return true;
    }
};

std::string projector_of(const Keys& k) {
    std::string p = k.str("clip.vision.projector_type");
    return p.empty() ? k.str("clip.projector_type") : p;
}

// Bicubic resize of float RGB (interleaved), separable, with the filter widened when shrinking so every source
// pixel counts (as PIL's and torchvision's antialiased bicubic do; a = -0.5).
std::vector<float> resize_bicubic(const std::vector<float>& src, int sw, int sh, int dw, int dh) {
    auto cubic = [](double x) {
        const double a = -0.5;
        x = std::fabs(x);
        if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
        if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    };
    struct Tap { int first = 0; std::vector<float> w; };
    auto taps = [&](int in, int out) {
        std::vector<Tap> t(out);
        const double scale = (double)in / out, support = 2.0 * std::max(1.0, scale), fs = std::max(1.0, scale);
        for (int i = 0; i < out; ++i) {
            const double center = (i + 0.5) * scale;
            int lo = (int)std::floor(center - support + 0.5), hi = (int)std::floor(center + support + 0.5);
            lo = std::max(lo, 0); hi = std::min(hi, in);
            double sum = 0;
            std::vector<double> w;
            for (int j = lo; j < hi; ++j) { w.push_back(cubic((j - center + 0.5) / fs)); sum += w.back(); }
            t[i].first = lo;
            for (double v : w) t[i].w.push_back((float)(sum != 0 ? v / sum : 0));
        }
        return t;
    };
    const auto tx = taps(sw, dw), ty = taps(sh, dh);
    std::vector<float> mid((size_t)dw * sh * 3), out((size_t)dw * dh * 3);
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < dw; ++x) {
            float acc[3] = {0, 0, 0};
            const Tap& t = tx[x];
            for (size_t k = 0; k < t.w.size(); ++k) {
                const float* p = &src[((size_t)y * sw + t.first + k) * 3];
                for (int c = 0; c < 3; ++c) acc[c] += t.w[k] * p[c];
            }
            for (int c = 0; c < 3; ++c) mid[((size_t)y * dw + x) * 3 + c] = acc[c];
        }
    for (int y = 0; y < dh; ++y) {
        const Tap& t = ty[y];
        for (int x = 0; x < dw; ++x) {
            float acc[3] = {0, 0, 0};
            for (size_t k = 0; k < t.w.size(); ++k) {
                const float* p = &mid[((size_t)(t.first + k) * dw + x) * 3];
                for (int c = 0; c < 3; ++c) acc[c] += t.w[k] * p[c];
            }
            for (int c = 0; c < 3; ++c) out[((size_t)y * dw + x) * 3 + c] = std::min(255.0f, std::max(0.0f, acc[c]));
        }
    }
    return out;
}

// One tensor of the file as floats (F32, F16 or BF16).
std::vector<float> read_floats(std::ifstream& f, gguf_context* g, const char* name, ggml_tensor* t) {
    const int64_t i = gguf_find_tensor(g, name);
    if (i < 0 || !t) throw std::runtime_error(std::string("vision file is missing ") + name);
    std::vector<char> raw(ggml_nbytes(t));
    f.seekg((std::streamoff)(gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i)));
    if (!f.read(raw.data(), (std::streamsize)raw.size())) throw std::runtime_error("vision file is cut short");
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) std::memcpy(out.data(), raw.data(), sizeof(float) * n);
    else if (t->type == GGML_TYPE_F16) for (int64_t k = 0; k < n; ++k) out[k] = ggml_fp16_to_fp32(((const ggml_fp16_t*)raw.data())[k]);
    else if (t->type == GGML_TYPE_BF16) for (int64_t k = 0; k < n; ++k) out[k] = ggml_bf16_to_fp32(((const ggml_bf16_t*)raw.data())[k]);
    else throw std::runtime_error(std::string("vision file: unexpected type for ") + name);
    return out;
}

}   // namespace

bool is_vision_file(const std::string& path, std::string* why) {
    gguf_init_params gp{ /*no_alloc=*/true, nullptr };
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    if (!g) { if (why) *why = "not a GGUF file"; return false; }
    Keys k{g};
    const bool clip = k.str("general.architecture") == "clip";
    const std::string p = projector_of(k);
    gguf_free(g);
    if (!clip) { if (why) *why = "not a vision (mmproj) file"; return false; }
    if (p != "qwen3vl_merger") { if (why) *why = "vision file of type '" + p + "', not supported yet"; return false; }
    return true;
}

Vision::~Vision() {
    if (sched_) ggml_backend_sched_free(sched_);
    if (sched_help_) ggml_backend_sched_free(sched_help_);
    if (help_) ggml_backend_free(help_);
    if (wbuf_) ggml_backend_buffer_free(wbuf_);
    if (xbuf_) ggml_backend_buffer_free(xbuf_);
    if (main_ && main_ != cpu_) ggml_backend_free(main_);
    if (cpu_) { forget_cpu_backend(cpu_); ggml_backend_free(cpu_); }
    if (wctx_) ggml_free(wctx_);
    if (xctx_) ggml_free(xctx_);
    if (gguf_) gguf_free(gguf_);
}

std::unique_ptr<Vision> Vision::load(const std::string& path, const std::string& device) {
    std::unique_ptr<Vision> v(new Vision());
    gguf_init_params gp{ /*no_alloc=*/true, &v->wctx_ };
    v->gguf_ = gguf_init_from_file(path.c_str(), gp);
    if (!v->gguf_) throw std::runtime_error("failed to read GGUF: " + path);
    gguf_context* g = v->gguf_;
    Keys k{g};
    if (k.str("general.architecture") != "clip") throw std::runtime_error("not a vision (mmproj) file");
    const std::string proj = projector_of(k);
    if (proj != "qwen3vl_merger") throw std::runtime_error("vision file of type '" + proj + "' is not supported yet");
    if (k.num("clip.has_vision_encoder", 1) == 0) throw std::runtime_error("the file has no vision encoder");
    v->n_embd_ = (int)k.num("clip.vision.embedding_length", 0);
    v->n_head_ = (int)k.num("clip.vision.attention.head_count", 0);
    v->n_ff_ = (int)k.num("clip.vision.feed_forward_length", 0);
    v->patch_ = (int)k.num("clip.vision.patch_size", 16);
    v->merge_ = (int)k.num("clip.vision.spatial_merge_size", 2);
    v->out_dim_ = (int)k.num("clip.vision.projection_dim", 0);
    v->eps_ = (float)k.num("clip.vision.attention.layer_norm_epsilon", 1e-6);
    const int n_block = (int)k.num("clip.vision.block_count", 0);
    k.floats("clip.vision.image_mean", v->mean_, 3);
    k.floats("clip.vision.image_std", v->std_, 3);
    const int unit = v->patch_ * v->merge_;                      // pixels per image word, per side
    const double minp = k.num("clip.vision.image_min_pixels", 0), maxp = k.num("clip.vision.image_max_pixels", 0);
    if (minp > 0) v->min_tokens_ = std::max(1, (int)(minp / (unit * unit)));
    if (maxp > 0) v->max_tokens_ = std::max(v->min_tokens_, (int)(maxp / (unit * unit)));
    if (v->n_embd_ <= 0 || v->n_head_ <= 0 || n_block <= 0 || v->n_embd_ % v->n_head_ || v->out_dim_ <= 0 || v->merge_ != 2)
        throw std::runtime_error("vision file with incomplete or unexpected settings");
    if (k.num("clip.vision.is_deepstack_layers", 0) != 0) {
        const int64_t id = gguf_find_key(g, "clip.vision.is_deepstack_layers");
        if (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
            const bool* d = (const bool*)gguf_get_arr_data(g, id);
            for (size_t i = 0; i < gguf_get_arr_n(g, id); ++i)
                if (d[i]) throw std::runtime_error("Qwen3-VL DeepStack vision files are not supported yet");
        }
    }

    auto T = [&](const std::string& name, bool need) -> ggml_tensor* {
        ggml_tensor* t = ggml_get_tensor(v->wctx_, name.c_str());
        if (!t && need) throw std::runtime_error("vision file is missing " + name);
        return t;
    };
    ggml_tensor* pw0 = T("v.patch_embd.weight", true);
    ggml_tensor* pw1 = T("v.patch_embd.weight.1", false);
    v->patch_b_ = T("v.patch_embd.bias", false);
    v->pos_ = T("v.position_embd.weight", true);
    v->pos_side_ = (int)std::lround(std::sqrt((double)v->pos_->ne[1]));
    if (v->pos_side_ * v->pos_side_ != v->pos_->ne[1] || v->pos_->ne[0] != v->n_embd_)
        throw std::runtime_error("vision file: unexpected position table");
    v->post_ln_ = T("v.post_ln.weight", true); v->post_ln_b_ = T("v.post_ln.bias", false);
    v->mm0_ = T("mm.0.weight", true); v->mm0_b_ = T("mm.0.bias", false);
    v->mm2_ = T("mm.2.weight", true); v->mm2_b_ = T("mm.2.bias", false);
    for (int il = 0; il < n_block; ++il) {
        const std::string b = "v.blk." + std::to_string(il) + ".";
        Block B;
        B.ln1 = T(b + "ln1.weight", true);       B.ln1_b = T(b + "ln1.bias", false);
        B.qkv = T(b + "attn_qkv.weight", true);  B.qkv_b = T(b + "attn_qkv.bias", false);
        B.out = T(b + "attn_out.weight", true);  B.out_b = T(b + "attn_out.bias", false);
        B.ln2 = T(b + "ln2.weight", true);       B.ln2_b = T(b + "ln2.bias", false);
        B.up = T(b + "ffn_up.weight", true);     B.up_b = T(b + "ffn_up.bias", false);
        B.down = T(b + "ffn_down.weight", true); B.down_b = T(b + "ffn_down.bias", false);
        v->blocks_.push_back(B);
    }
    for (ggml_tensor* t = ggml_get_first_tensor(v->wctx_); t; t = ggml_get_next_tensor(v->wctx_, t)) v->weight_bytes_ += ggml_nbytes(t);

    // ---- the device: a GPU with room to spare, else the CPU ----
    load_backends();
    v->cpu_ = cpu_backend_init();
    if (!v->cpu_) throw std::runtime_error("failed to start the CPU backend");
    v->main_ = v->cpu_;
    ggml_backend_dev_t dev = nullptr;
    static const bool force_ram = std::getenv("TIERLM_VISION_RAM") != nullptr;   // tests: as if no GPU had room for it
    // "auto:<device>": the chat model's GPU (as "CUDA0"): the image reader goes on it when there's room, else stays in
    // RAM with that GPU helping. Never another, slower chip (a laptop's built-in graphics would be chosen otherwise:
    // it has shared memory to spare, but reads an image slower than the CPU with the real GPU helping).
    ggml_backend_dev_t pref = nullptr;
    std::string mode = device;
    if (device.rfind("auto:", 0) == 0) {
        pref = ggml_backend_dev_by_name(device.substr(5).c_str());
        mode = "auto";
    }
    const size_t need = v->weight_bytes_ + ((size_t)1536 << 20);              // weights + working memory for a page
    auto room = [&](ggml_backend_dev_t d) { size_t free = 0, total = 0; ggml_backend_dev_memory(d, &free, &total); return free; };
    // Without a chat-model GPU: the best discrete GPU; built-in graphics only when there is no discrete one.
    auto best_gpu = [&](bool need_room) -> ggml_backend_dev_t {
        bool any_discrete = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) any_discrete |= ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
        ggml_backend_dev_t best = nullptr;
        size_t most = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const auto t = ggml_backend_dev_type(d);
            if (t != GGML_BACKEND_DEVICE_TYPE_GPU && !(t == GGML_BACKEND_DEVICE_TYPE_IGPU && !any_discrete)) continue;
            const size_t f = room(d);
            if ((!need_room || f >= need) && f > most) { best = d; most = f; }
        }
        return best;
    };
    if (mode == "auto" || mode == "gpu") {
        if (!force_ram) dev = pref ? (room(pref) >= need ? pref : nullptr) : best_gpu(true);
    } else if (mode != "cpu") {
        dev = ggml_backend_dev_by_name(mode.c_str());
        if (!dev) throw std::runtime_error("unknown device '" + mode + "'");
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) dev = nullptr;
    }
    if (dev) {
        v->main_ = ggml_backend_dev_init(dev, nullptr);
        if (!v->main_) throw std::runtime_error(std::string("failed to start ") + ggml_backend_dev_name(dev));
        v->device_name_ = std::string(ggml_backend_dev_name(dev)) + " (" + ggml_backend_dev_description(dev) + ")";
    } else v->device_name_ = "CPU";
    // No room for the weights on a GPU (the chat model fills it): they stay in RAM, but a GPU with a little room
    // still does the heavy work, the weights copied over op by op (ggml's offload of big batches). An image is
    // thousands of patches, so that is far faster than the CPU (seconds instead of a minute). Without room even for
    // that, the CPU does it all (encode() falls back by itself).
    if (!dev && mode == "auto" && !std::getenv("TIERLM_VISION_NO_HELP")) {
        ggml_backend_dev_t h = pref ? pref : best_gpu(false);
        if (h && (v->help_ = ggml_backend_dev_init(h, nullptr)))
            v->device_name_ = std::string("CPU + ") + ggml_backend_dev_name(h) + " (" + ggml_backend_dev_description(h) + ")";
    }

    // ---- the weights. The patch layer is a 3D convolution over two copies of the image (Qwen treats a
    // image as a 2-frame video); with both frames the same, that is one 2D convolution with the two halves of
    // the kernel added, kept here as a plain matrix [3*16*16, n_embd]. ----
    {
        std::lock_guard<std::recursive_mutex> lk(device_mutex());
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        std::vector<float> pw = read_floats(f, g, "v.patch_embd.weight", pw0);
        if (pw1) {
            const std::vector<float> b = read_floats(f, g, "v.patch_embd.weight.1", pw1);
            if (b.size() != pw.size()) throw std::runtime_error("vision file: patch weights don't match");
            for (size_t i = 0; i < pw.size(); ++i) pw[i] += b[i];
        }
        const int pin = 3 * v->patch_ * v->patch_;
        if ((int64_t)pw.size() != (int64_t)pin * v->n_embd_) throw std::runtime_error("vision file: unexpected patch weights");
        ggml_init_params xp{ ggml_tensor_overhead() * 4, nullptr, true };
        v->xctx_ = ggml_init(xp);
        v->patch_w_ = ggml_new_tensor_2d(v->xctx_, GGML_TYPE_F32, pin, v->n_embd_);
        ggml_set_name(v->patch_w_, "v.patch_embd.sum");
        v->xbuf_ = ggml_backend_alloc_ctx_tensors(v->xctx_, v->main_);
        if (!v->xbuf_) throw std::runtime_error("not enough memory for the vision model");
        ggml_backend_buffer_set_usage(v->xbuf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);   // (so a helping GPU may take it)
        ggml_backend_tensor_set(v->patch_w_, pw.data(), 0, sizeof(float) * pw.size());

        // everything else straight from the file (the two patch halves too: small, and not used)
        v->wbuf_ = ggml_backend_alloc_ctx_tensors(v->wctx_, v->main_);
        if (!v->wbuf_) throw std::runtime_error("not enough memory for the vision model");
        ggml_backend_buffer_set_usage(v->wbuf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        const size_t data0 = gguf_get_data_offset(g);
        std::vector<char> buf;
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            ggml_tensor* t = ggml_get_tensor(v->wctx_, gguf_get_tensor_name(g, i));
            if (!t) continue;
            buf.resize(ggml_nbytes(t));
            f.seekg((std::streamoff)(data0 + gguf_get_tensor_offset(g, i)));
            if (!f.read(buf.data(), (std::streamsize)buf.size())) throw std::runtime_error("vision file is cut short");
            ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
        }
    }
    std::vector<ggml_backend_t> backends;
    if (v->main_ != v->cpu_) backends.push_back(v->main_);
    backends.push_back(v->cpu_);
    v->sched_ = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), 16384, false, true);
    if (!v->sched_) throw std::runtime_error("failed to create the scheduler");
    if (v->help_) {
        ggml_backend_t hb[2] = { v->help_, v->cpu_ };
        v->sched_help_ = ggml_backend_sched_new(hb, nullptr, 2, 16384, false, true);   // op_offload: big ops to the GPU
    }
    return v;
}

void Vision::resized(int width, int height, int max_tokens, int& rw, int& rh) const {
    const int unit = patch_ * merge_;
    const double maxp = (double)(max_tokens > 0 ? std::min(max_tokens, max_tokens_) : max_tokens_) * unit * unit;
    const double minp = (double)min_tokens_ * unit * unit;
    // Qwen's smart resize: the nearest multiples of 32, then scaled down (or up) to the pixel limits.
    const double h = height, w = width;
    rh = std::max(unit, (int)std::lround(h / unit) * unit); rw = std::max(unit, (int)std::lround(w / unit) * unit);
    if ((double)rh * rw > maxp) {
        const double beta = std::sqrt(h * w / maxp);
        rh = std::max(unit, (int)std::floor(h / beta / unit) * unit);
        rw = std::max(unit, (int)std::floor(w / beta / unit) * unit);
    } else if ((double)rh * rw < minp) {
        const double beta = std::sqrt(minp / (h * w));
        rh = (int)std::ceil(h * beta / unit) * unit;
        rw = (int)std::ceil(w * beta / unit) * unit;
    }
}

int Vision::tokens_for(int width, int height, int max_tokens) const {
    if (width <= 0 || height <= 0) return 0;
    int rw = 0, rh = 0;
    resized(width, height, max_tokens, rw, rh);
    return (rw / (patch_ * merge_)) * (rh / (patch_ * merge_));
}

Vision::Prepared Vision::prepare(const Image& img, int max_tokens) const {
    if (img.width <= 0 || img.height <= 0 || img.rgb.size() != (size_t)img.width * img.height * 3)
        throw std::runtime_error("empty image");
    if (std::max(img.width, img.height) > 200 * std::min(img.width, img.height))
        throw std::runtime_error("image is too long and thin (more than 200:1)");
    int rw = 0, rh = 0;
    resized(img.width, img.height, max_tokens, rw, rh);
    std::vector<float> src(img.rgb.begin(), img.rgb.end());
    const std::vector<float> px = (rh == img.height && rw == img.width) ? src : resize_bicubic(src, img.width, img.height, rw, rh);
    Prepared p;
    p.width = rw; p.height = rh; p.gh = rh / patch_; p.gw = rw / patch_;
    const int P = patch_, M = merge_, per = 3 * P * P;
    p.patches.resize((size_t)p.gh * p.gw * per);
    // Order: merge blocks row by row; inside a block its 2x2 patches row by row; inside a patch channel, row,
    // column (the layout of the patch layer's weights).
    size_t n = 0;
    for (int by = 0; by < p.gh / M; ++by)
        for (int bx = 0; bx < p.gw / M; ++bx)
            for (int iy = 0; iy < M; ++iy)
                for (int ix = 0; ix < M; ++ix) {
                    float* out = &p.patches[n++ * per];
                    const int y0 = (by * M + iy) * P, x0 = (bx * M + ix) * P;
                    for (int c = 0; c < 3; ++c)
                        for (int y = 0; y < P; ++y)
                            for (int x = 0; x < P; ++x)
                                out[(c * P + y) * P + x] = (px[((size_t)(y0 + y) * rw + x0 + x) * 3 + c] / 255.0f - mean_[c]) / std_[c];
                }
    return p;
}

std::vector<float> Vision::encode(const Prepared& p, int n_threads) {
    const int N = p.gh * p.gw, E = n_embd_, H = n_head_, hd = E / H, M = merge_, per = 3 * patch_ * patch_;
    if (N <= 0 || (int)p.patches.size() != N * per) throw std::runtime_error("image not prepared");
    // Where each patch is (row, column), in the order of the patches.
    std::vector<int32_t> rows(N), cols(N);
    {
        int n = 0;
        for (int by = 0; by < p.gh / M; ++by)
            for (int bx = 0; bx < p.gw / M; ++bx)
                for (int iy = 0; iy < M; ++iy)
                    for (int ix = 0; ix < M; ++ix) { rows[n] = by * M + iy; cols[n] = bx * M + ix; ++n; }
    }
    // The learned 48x48 position table, resampled to this grid (bilinear, corners aligned): 4 rows and weights
    // per patch.
    std::vector<int32_t> pidx(4 * (size_t)N);
    std::vector<float> pw(4 * (size_t)N);
    {
        const int S = pos_side_;
        auto axis = [&](int i, int size, int& a, int& b, float& wa, float& wb) {
            const double src = size > 1 ? (double)i * (S - 1) / (size - 1) : 0.0;
            const int f = (int)std::floor(src);
            const double d = src - f;
            a = std::min(f, S - 1); b = std::min(f + 1, S - 1);
            wa = (float)(1 - d); wb = (float)d;
        };
        for (int n = 0; n < N; ++n) {
            int r0, r1, c0, c1; float wr0, wr1, wc0, wc1;
            axis(rows[n], p.gh, r0, r1, wr0, wr1);
            axis(cols[n], p.gw, c0, c1, wc0, wc1);
            const int32_t id[4] = { r0 * S + c0, r0 * S + c1, r1 * S + c0, r1 * S + c1 };
            const float w[4] = { wr0 * wc0, wr0 * wc1, wr1 * wc0, wr1 * wc1 };
            for (int k = 0; k < 4; ++k) { pidx[(size_t)k * N + n] = id[k]; pw[(size_t)k * N + n] = w[k]; }
        }
    }
    // 2D rotary positions: [rows][columns][0][0], as ggml's vision rope reads them.
    std::vector<int32_t> rpos(4 * (size_t)N, 0);
    for (int n = 0; n < N; ++n) { rpos[n] = rows[n]; rpos[N + n] = cols[n]; }

    const size_t GS = 16384;
    ggml_init_params ip{ ggml_tensor_overhead() * GS + ggml_graph_overhead_custom(GS, false), nullptr, true };
    ggml_context* c = ggml_init(ip);
    ggml_tensor* inp = ggml_new_tensor_2d(c, GGML_TYPE_F32, per, N); ggml_set_input(inp);
    ggml_tensor* pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * N); ggml_set_input(pos);
    ggml_tensor* tap_idx[4]; ggml_tensor* tap_w[4];
    for (int k = 0; k < 4; ++k) {
        tap_idx[k] = ggml_new_tensor_1d(c, GGML_TYPE_I32, N); ggml_set_input(tap_idx[k]);
        tap_w[k] = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, N); ggml_set_input(tap_w[k]);
    }
    auto norm = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
        x = ggml_mul(c, ggml_norm(c, x, eps_), w);
        return b ? ggml_add(c, x, b) : x;
    };
    auto lin = [&](ggml_tensor* w, ggml_tensor* b, ggml_tensor* x) {
        ggml_tensor* y = ggml_mul_mat(c, w, x);
        return b ? ggml_add(c, y, b) : y;
    };
    // GELU in its tanh form, written out: ggml_gelu reads a 16-bit table on the CPU, close but not exact.
    auto gelu_tanh = [&](ggml_tensor* x) {
        ggml_tensor* inner = ggml_scale(c, ggml_add(c, x, ggml_scale(c, ggml_mul(c, ggml_sqr(c, x), x), 0.044715f)), 0.7978845608028654f);
        return ggml_mul(c, ggml_scale(c, x, 0.5f), ggml_scale_bias(c, ggml_tanh(c, inner), 1.0f, 1.0f));
    };
    ggml_tensor* x = lin(patch_w_, patch_b_, inp);                                 // [E, N]
    for (int k = 0; k < 4; ++k) x = ggml_add(c, x, ggml_mul(c, ggml_get_rows(c, pos_, tap_idx[k]), tap_w[k]));
    const float scale = 1.0f / std::sqrt((float)hd);
    static const bool exact = std::getenv("TIERLM_VISION_EXACT") != nullptr;   // tests: the plain attention always
    const bool use_fa = !exact && N > 1024;
    int sections[GGML_MROPE_SECTIONS] = { hd / 4, hd / 4, 0, 0 };
    for (const Block& B : blocks_) {
        ggml_tensor* h = norm(x, B.ln1, B.ln1_b);
        ggml_tensor* qkv = lin(B.qkv, B.qkv_b, h);                                 // [3E, N]: q, k, v
        ggml_tensor* q = ggml_view_3d(c, qkv, hd, H, N, (size_t)hd * ggml_element_size(qkv), qkv->nb[1], 0);
        ggml_tensor* k = ggml_view_3d(c, qkv, hd, H, N, (size_t)hd * ggml_element_size(qkv), qkv->nb[1], (size_t)E * ggml_element_size(qkv));
        ggml_tensor* v = ggml_view_3d(c, qkv, hd, H, N, (size_t)hd * ggml_element_size(qkv), qkv->nb[1], (size_t)2 * E * ggml_element_size(qkv));
        q = ggml_rope_multi(c, ggml_cont(c, q), pos, nullptr, hd / 2, sections, GGML_ROPE_TYPE_VISION, 32768, rope_base_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_multi(c, ggml_cont(c, k), pos, nullptr, hd / 2, sections, GGML_ROPE_TYPE_VISION, 32768, rope_base_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        // Attention over the whole image (no mask). Flash attention keeps memory small for big images (a
        // full page is ~16,000 patches: the plain way would need N*N scores per head); the plain way is exact,
        // for small images and tests.
        ggml_tensor* qp = ggml_permute(c, q, 0, 2, 1, 3);                          // [hd, N, H]
        ggml_tensor* att;
        if (use_fa) {
            ggml_tensor* kp = ggml_cast(c, ggml_permute(c, k, 0, 2, 1, 3), GGML_TYPE_F16);
            ggml_tensor* vp = ggml_cast(c, ggml_permute(c, ggml_cont(c, v), 0, 2, 1, 3), GGML_TYPE_F16);
            att = ggml_flash_attn_ext(c, qp, kp, vp, nullptr, scale, 0.0f, 0.0f);  // [hd, H, N]
            ggml_prec_set_acc(att, GGML_PREC_F32);
            att = ggml_reshape_2d(c, att, E, N);
        } else {
            ggml_tensor* kp = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
            ggml_tensor* kq = ggml_soft_max_ext(c, ggml_mul_mat(c, kp, qp), nullptr, scale, 0.0f);   // [N_k, N_q, H]
            ggml_tensor* vp = ggml_cont(c, ggml_permute(c, ggml_cont(c, v), 1, 2, 0, 3));   // [N, hd, H]
            att = ggml_mul_mat(c, vp, kq);                                          // [hd, N_q, H]
            att = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, att, 0, 2, 1, 3)), E, N);
        }
        x = ggml_add(c, x, lin(B.out, B.out_b, att));
        h = norm(x, B.ln2, B.ln2_b);
        h = lin(B.down, B.down_b, gelu_tanh(lin(B.up, B.up_b, h)));             // GELU (tanh form)
        x = ggml_add(c, x, h);
    }
    // Merge: LayerNorm, each 2x2 block (4 consecutive patches) into one vector, then the MLP (exact GELU).
    x = norm(x, post_ln_, post_ln_b_);
    x = ggml_reshape_2d(c, ggml_cont(c, x), E * M * M, N / (M * M));
    x = lin(mm2_, mm2_b_, ggml_gelu_erf(c, lin(mm0_, mm0_b_, x)));               // [out_dim, N/4]
    ggml_set_output(x);
    ggml_cgraph* gf = ggml_new_graph_custom(c, GS, false);
    ggml_build_forward_expand(gf, x);

    std::vector<float> out((size_t)out_dim_ * (N / (M * M)));
    {
        std::lock_guard<std::recursive_mutex> lk(device_mutex());
        // With a helping GPU: its working memory may not fit next to the chat model; then the CPU alone.
        ggml_backend_sched_t sched = nullptr;
        if (sched_help_) {
            ggml_backend_sched_reset(sched_help_);
            if (ggml_backend_sched_alloc_graph(sched_help_, gf)) sched = sched_help_;
            else {
                std::fprintf(stderr, "vision: no room on the GPU to help with this image: the CPU reads it\n");
                ggml_backend_sched_reset(sched_help_);
            }
        }
        if (!sched) {
            sched = sched_;
            ggml_backend_sched_reset(sched);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) { ggml_free(c); throw std::runtime_error("not enough memory to read this image"); }
        }
        last_helped_ = sched == sched_help_;
        ggml_backend_tensor_set(inp, p.patches.data(), 0, sizeof(float) * p.patches.size());
        ggml_backend_tensor_set(pos, rpos.data(), 0, sizeof(int32_t) * rpos.size());
        for (int k = 0; k < 4; ++k) {
            ggml_backend_tensor_set(tap_idx[k], pidx.data() + (size_t)k * N, 0, sizeof(int32_t) * N);
            ggml_backend_tensor_set(tap_w[k], pw.data() + (size_t)k * N, 0, sizeof(float) * N);
        }
        set_n_threads(cpu_, n_threads);
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) { ggml_free(c); throw std::runtime_error("reading the image failed"); }
        ggml_backend_tensor_get(x, out.data(), 0, sizeof(float) * out.size());
    }
    ggml_free(c);
    return out;
}

}
