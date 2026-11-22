// tierlm-attn-check (developers): ggml's attention when it is told how much each query can see, against the same
// attention going through the whole cache behind its mask, on this build's GPU. Run it after changing the ggml pin.
//
// Flash-Next's sparse attention (QSA) lets a query past the budget see the budget's worth of whole blocks and its own
// unfinished block; tierlm gives the attention op that bound (ggml_flash_attn_ext_set_n_kv_max) and the GPU then
// gathers just those keys and values. This makes a cache of the model's shape (head size 256, 2 key/value heads, 24
// heads, blocks of 4, budget 2,048) full of random numbers, with masks like QSA's, and for steps of several sizes
// prints the largest difference between the two results and what each takes.
//
//   tierlm-attn-check [positions in the cache = 73728] [cache type: q8_0 or f16] [steps' sizes, e.g. 1,4,9,256]
// TIERLM_DEVICE: the device (default: the first GPU).
#include "backends.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tierlm;

int main(int argc, char** argv) {
    try {
        load_backends();
        const int n_kv = argc > 1 ? std::max(4096, std::atoi(argv[1]) / 256 * 256) : 73728;
        const std::string tname = argc > 2 ? argv[2] : "q8_0";
        const ggml_type kvt = tname == "f16" ? GGML_TYPE_F16 : tname == "f32" ? GGML_TYPE_F32 : GGML_TYPE_Q8_0;
        std::vector<int> sizes = {1, 2, 3, 4, 9, 64, 256, 2048};
        if (argc > 3) {
            sizes.clear();
            std::stringstream ss(argv[3]);
            for (std::string s; std::getline(ss, s, ',');) sizes.push_back(std::max(1, std::atoi(s.c_str())));
        }
        const int HD = 256, HKV = 2, NH = 24, R = 4, BUDGET = 2048;

        ggml_backend_dev_t dev = nullptr;
        if (const char* want = std::getenv("TIERLM_DEVICE")) dev = ggml_backend_dev_by_name(want);
        for (size_t i = 0; !dev && i < ggml_backend_dev_count(); ++i)
            if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) dev = ggml_backend_dev_get(i);
        if (!dev) throw std::runtime_error("no GPU here (TIERLM_DEVICE names another device)");
        ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!gpu || !cpu) throw std::runtime_error("could not start the device");
        std::printf("%s (%s), a cache of %d positions as %s, head size %d, %d key/value heads, %d heads\n", ggml_backend_dev_name(dev),
                    ggml_backend_dev_description(dev), n_kv, ggml_type_name(kvt), HD, HKV, NH);

        // the cache, as tierlm keeps it: [head size, key/value heads, positions]
        ggml_init_params wp{ 8 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
        ggml_context* wctx = ggml_init(wp);
        ggml_tensor* kc = ggml_new_tensor_3d(wctx, kvt, HD, HKV, n_kv);
        ggml_tensor* vc = ggml_new_tensor_3d(wctx, kvt, HD, HKV, n_kv);
        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, gpu);
        if (!wbuf) throw std::runtime_error("no room for the cache on the device");
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        {
            const size_t n = (size_t)HD * HKV * n_kv;
            std::vector<float> f(n);
            for (ggml_tensor* t : {kc, vc}) {
                for (float& x : f) x = nd(rng);
                if (kvt == GGML_TYPE_F32) ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
                else if (kvt == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> h(n);
                    ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t)n);
                    ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
                } else {
                    std::vector<uint8_t> q(ggml_nbytes(t));
                    ggml_quantize_chunk(kvt, f.data(), q.data(), 0, (int64_t)HKV * n_kv, HD, nullptr);
                    ggml_backend_tensor_set(t, q.data(), 0, ggml_nbytes(t));
                }
            }
        }
        const bool on_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;   // (TIERLM_DEVICE=CPU: the op's own check)
        ggml_backend_t backends[2] = { gpu, cpu };
        const int n_backends = on_cpu ? 1 : 2;
        bool all_ok = true;
        for (int n_tok : sizes) {
            if (n_tok > n_kv / 2) continue;
            // masks like QSA's for a conversation that fills the cache, the step being its last n_tok positions
            const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
            std::vector<ggml_fp16_t> mask((size_t)n_kv * n_tok, ninf);
            std::vector<int> picked;
            int most = 0;
            for (int t = 0; t < n_tok; ++t) {
                const int pos = n_kv - n_tok + t, whole = (pos + 1) / R, top = BUDGET / R;
                if (picked.empty()) {                              // the first query's blocks
                    std::vector<int> all(whole);
                    for (int b = 0; b < whole; ++b) all[b] = b;
                    std::shuffle(all.begin(), all.end(), rng);
                    picked.assign(all.begin(), all.begin() + std::min(top, whole));
                } else {                                           // the next: most of the same, a few others
                    for (int k = 0; k < top / 16 && k < (int)picked.size(); ++k) {
                        const int nb = (int)(rng() % (uint32_t)whole);
                        if (std::find(picked.begin(), picked.end(), nb) == picked.end()) picked[rng() % picked.size()] = nb;
                    }
                }
                ggml_fp16_t* row = mask.data() + (size_t)t * n_kv;
                int seen = 0;
                for (int b : picked) for (int j = 0; j < R; ++j) { row[b * R + j] = zero; ++seen; }
                for (int p = whole * R; p <= pos; ++p) { row[p] = zero; ++seen; }
                most = std::max(most, seen);
            }
            std::vector<float> qv((size_t)HD * NH * n_tok);
            for (float& x : qv) x = nd(rng);

            auto run = [&](bool sparse, std::vector<float>& out) -> double {
                ggml_init_params gp{ 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
                ggml_context* c = ggml_init(gp);
                ggml_cgraph* gf = ggml_new_graph(c);
                ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_F32, HD, NH, n_tok);  ggml_set_input(q);
                ggml_tensor* m = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv, n_tok);     ggml_set_input(m);
                ggml_tensor* Q = ggml_permute(c, q, 0, 2, 1, 3);
                ggml_tensor* K = ggml_permute(c, kc, 0, 2, 1, 3);
                ggml_tensor* V = ggml_permute(c, vc, 0, 2, 1, 3);
                ggml_tensor* o = ggml_flash_attn_ext(c, Q, K, V, m, 1.0f / std::sqrt((float)HD), 0.0f, 0.0f);
                if (sparse) ggml_flash_attn_ext_set_n_kv_max(o, BUDGET + R);
                ggml_set_output(o);
                ggml_build_forward_expand(gf, o);
                ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_backends, 256, false, true);
                if (!ggml_backend_sched_alloc_graph(sched, gf)) throw std::runtime_error("no room for the step's working memory");
                ggml_backend_tensor_set(q, qv.data(), 0, ggml_nbytes(q));
                ggml_backend_tensor_set(m, mask.data(), 0, ggml_nbytes(m));
                double best = 1e30;
                for (int it = 0; it < 6; ++it) {
                    const auto t0 = std::chrono::steady_clock::now();
                    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("the step failed");
                    ggml_backend_sched_synchronize(sched);
                    if (it > 0) best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                }
                out.resize(ggml_nelements(o));
                ggml_backend_tensor_get(o, out.data(), 0, ggml_nbytes(o));
                const bool on_gpu = ggml_backend_sched_get_tensor_backend(sched, o) == gpu;
                ggml_backend_sched_free(sched);
                ggml_free(c);
                return on_gpu ? best : -best;
            };
            std::vector<float> a, b;
            const double td = run(false, a), ts = run(true, b);
            double worst = 0.0, big = 0.0;
            bool finite = true;
            for (size_t i = 0; i < a.size(); ++i) {
                if (!std::isfinite(a[i]) || !std::isfinite(b[i])) finite = false;
                worst = std::max(worst, (double)std::fabs(a[i] - b[i]));
                big = std::max(big, (double)std::fabs(a[i]));
            }
            const bool ok = finite && big > 0 && worst <= 2e-3 * big;
            all_ok = all_ok && ok;
            std::printf("a step of %4d: each query sees up to %d positions; told so %.2f ms, behind the mask %.2f ms%s; "
                        "largest difference %.1e of the largest value  ->  %s\n", n_tok, most, std::fabs(ts), std::fabs(td),
                        td < 0 || ts < 0 ? " (not on the GPU)" : "", ok ? "ok" : "DIFFERENT");
            std::fflush(stdout);
        }
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        std::printf(all_ok ? "PASS\n" : "FAIL\n");
        return all_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
