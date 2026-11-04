// tierlm-bench: generation speed at different context lengths.
//   tierlm-bench model.gguf [-d auto|cpu|gpu] [-t threads] [contexts...]     e.g. tierlm-bench m.gguf 64 512 1024 2048
// For each context length: fill the cache with that many tokens, then time 32 single-token steps.
//   --parallel 1,2,4,8: instead, that many conversations at once (each at the first context length), one token
//   each per step: the total and per-conversation speed.
#include "model.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace tierlm;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s model.gguf [-d auto|cpu|gpu] [-t threads] [contexts...]\n", argv[0]); return 1; }
    std::string device = "auto";
    int threads = std::getenv("TIERLM_THREADS") ? std::atoi(std::getenv("TIERLM_THREADS")) : std::max(1, (int)std::thread::hardware_concurrency() / 2);
    std::vector<int> ctxs, pars;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--parallel") && i + 1 < argc) {
            for (char* p = argv[++i]; *p;) { pars.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
        }
        else if (!std::strcmp(argv[i], "-d") && i + 1 < argc) device = argv[++i];
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) threads = std::atoi(argv[++i]);
        else ctxs.push_back(std::atoi(argv[i]));
    }
    if (ctxs.empty()) ctxs = {64, 512, 1024, 2048};
    const int steps = 32;
    try {
        if (!pars.empty()) {
            const int P = *std::max_element(pars.begin(), pars.end()), ctx = ctxs[0];
            LoadOptions o;
            o.device = device;
            o.parallel = P;
            o.max_ctx = P * (ctx + steps + 8);
            auto m = Model::load(argv[1], o);
            std::printf("device: %s | KV cache %s | %d CPU threads | context %d each\n", m->device_name().c_str(), m->kv_type(), threads, ctx);
            m->eval(std::vector<int32_t>(16, 1), threads);
            for (int n : pars) {
                std::vector<Model::SeqInput> in;
                for (int s = 0; s < P; ++s) m->seq_reset(s);
                for (int s = 0; s < n; ++s) in.push_back({s, std::vector<int32_t>(ctx, 1 + s)});
                m->eval_seqs(in, threads);
                for (auto& x : in) x.ids = {7};
                m->eval_seqs(in, threads);                     // warm-up at this batch size
                auto t0 = Clock::now();
                for (int k = 0; k < steps; ++k) m->eval_seqs(in, threads);
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / steps;
                std::printf("  %2d at once: %6.1f ms/step  %6.1f tok/s total, %5.1f each\n", n, ms, 1000.0 * n / ms, 1000.0 / ms);
            }
            return 0;
        }
        auto m = Model::load(argv[1], *std::max_element(ctxs.begin(), ctxs.end()) + steps + 8, device);
        std::printf("device: %s | KV cache %s | flash attention %s | %d CPU threads\n",
                    m->device_name().c_str(), m->kv_type(), m->flash_attn() ? "on" : "off", threads);
        m->eval(std::vector<int32_t>(16, 1), threads);   // warm-up (first use compiles GPU pipelines)
        for (int ctx : ctxs) {
            m->reset();
            m->eval(std::vector<int32_t>(ctx, 1), threads);
            auto t0 = Clock::now();
            for (int s = 0; s < steps; ++s) m->eval({1}, threads);
            double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / steps;
            std::printf("  context %5d: %6.1f ms/token  (%5.1f tok/s)\n", ctx, ms, 1000.0 / ms);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
