#include "linktest.h"
#include "backends.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

namespace tierlm {


LinkTest link_test(const std::string& want) {
    using clk = std::chrono::steady_clock;
    const auto t_start = clk::now();
    LinkTest r;
    load_backends();
    // The GPU whose name is in `want` ("CUDA0 (NVIDIA GeForce RTX 5090)" names CUDA0, not the Vulkan view of the
    // same card), else whose description is, else the first GPU
    ggml_backend_dev_t dev = nullptr, by_desc = nullptr, first = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto ty = ggml_backend_dev_type(d);
        if (ty != GGML_BACKEND_DEVICE_TYPE_GPU && ty != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        const std::string name = ggml_backend_dev_name(d), desc = ggml_backend_dev_description(d);
        if (!first) first = d;
        if (!want.empty() && want.find(name) != std::string::npos) { dev = d; break; }
        if (!by_desc && !want.empty() && !desc.empty() && (want.find(desc) != std::string::npos || desc.find(want) != std::string::npos)) by_desc = d;
    }
    if (!dev) dev = by_desc ? by_desc : first;
    if (!dev) { r.error = "no GPU found"; return r; }
    r.device = std::string(ggml_backend_dev_name(dev)) + " (" + ggml_backend_dev_description(dev) + ")";
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) { r.error = "couldn't start the GPU"; return r; }

    ggml_backend_buffer_t dbuf = nullptr;
    for (size_t mb : {512, 256, 128, 64}) {                 // (as much as the card has free)
        dbuf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), mb << 20);
        if (dbuf) { r.big = mb << 20; break; }
    }
    if (!dbuf) { ggml_backend_free(be); r.error = "not enough free memory on the card (64 MB)"; return r; }
    ggml_init_params ip{ggml_tensor_overhead() * 2, nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t)r.big);
    ggml_backend_tensor_alloc(dbuf, t, ggml_backend_buffer_get_base(dbuf));

    ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(dev);
    ggml_backend_buffer_t hbuf = hbt ? ggml_backend_buft_alloc_buffer(hbt, r.big) : nullptr;
    char* pin = hbuf ? (char*)ggml_backend_buffer_get_base(hbuf) : nullptr;
    r.pinned = pin != nullptr;
    std::vector<char> plain(r.big, 1);
    if (pin) std::memset(pin, 1, r.big);

    auto secs = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
    auto best = [&](int reps, auto&& run) {
        double b = 1e30;
        for (int i = 0; i < reps; ++i) { const auto a = clk::now(); run(); ggml_backend_synchronize(be); b = std::min(b, secs(a)); }
        return b;
    };
    const double gb = (double)r.big / 1e9;
    if (pin) {
        best(1, [&] { ggml_backend_tensor_set(t, pin, 0, r.big); });                       // (warm up)
        r.to_card = gb / best(5, [&] { ggml_backend_tensor_set(t, pin, 0, r.big); });
        r.to_ram  = gb / best(5, [&] { ggml_backend_tensor_get(t, pin, 0, r.big); });
        const size_t piece = 2 << 20, n = r.big / piece;
        r.pieces_each = (double)(piece * n) / 1e9 / best(3, [&] {
            for (size_t i = 0; i < n; ++i) { ggml_backend_tensor_set_async(be, t, pin + i * piece, i * piece, piece); ggml_backend_synchronize(be); }
        });
        r.pieces_queued = (double)(piece * n) / 1e9 / best(3, [&] {
            for (size_t i = 0; i < n; ++i) ggml_backend_tensor_set_async(be, t, pin + i * piece, i * piece, piece);
        });
    }
    r.to_card_plain = gb / best(3, [&] { ggml_backend_tensor_set(t, plain.data(), 0, r.big); });

    if (hbuf) ggml_backend_buffer_free(hbuf);
    ggml_free(ctx);
    ggml_backend_buffer_free(dbuf);
    ggml_backend_free(be);
    r.ok = true;
    r.seconds = secs(t_start);
    return r;
}

} // namespace tierlm
