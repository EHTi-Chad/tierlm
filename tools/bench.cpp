// tierlm-bench: generation speed at different context lengths.
//   tierlm-bench model.gguf [-d auto|cpu|gpu] [-t threads] [contexts...]     e.g. tierlm-bench m.gguf 64 512 1024 2048
// For each context length: fill the cache with that many tokens, then time 32 single-token steps.
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
    std::vector<int> ctxs;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-d") && i + 1 < argc) device = argv[++i];
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) threads = std::atoi(argv[++i]);
        else ctxs.push_back(std::atoi(argv[i]));
    }
    if (ctxs.empty()) ctxs = {64, 512, 1024, 2048};
    const int steps = 32;
    try {
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
