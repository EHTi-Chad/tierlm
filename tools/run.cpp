// tierlm-run: tokenize a prompt, run OLMoE, and generate text greedily.
#include "olmoe.h"
#include "tokenizer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <thread>

using namespace tierlm;
using Clock = std::chrono::steady_clock;
static double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s model.gguf \"prompt\" [-n tokens] [-t threads] [-c context]\n", argv[0]);
        return 1;
    }
    int n_gen = 32, n_threads = std::max(1, (int)std::thread::hardware_concurrency() / 2), n_ctx = 2048;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "-n")) n_gen = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "-t")) n_threads = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "-c")) n_ctx = std::atoi(argv[i + 1]);
    }
    try {
        auto t0 = Clock::now();
        Tokenizer tok = Tokenizer::from_gguf(argv[1]);
        auto model = OlmoeModel::load(argv[1], n_ctx);
        const auto& hp = model->hparams();
        std::fprintf(stderr, "loaded in %.1fs | %d layers, %d experts (%d active), vocab %d, ctx %d, %d threads\n",
                     since(t0), hp.n_layer, hp.n_expert, hp.n_expert_used, hp.n_vocab, model->max_ctx(), n_threads);

        // Prompt processing: the whole prompt in one call fills the KV cache.
        std::vector<int32_t> prompt = tok.encode(argv[2]);
        auto p0 = Clock::now();
        std::vector<float> lg = model->eval(prompt, n_threads);
        const double prompt_s = since(p0);

        // What the model was considering for the very first token.
        std::vector<int32_t> order(lg.size());
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                          [&](int32_t a, int32_t b) { return lg[a] > lg[b]; });
        double z = 0;
        for (float v : lg) z += std::exp((double)v - lg[order[0]]);
        std::fprintf(stderr, "prompt: %zu tokens in %.2fs\ntop 5 next tokens:", prompt.size(), prompt_s);
        for (int k = 0; k < 5; ++k)
            std::fprintf(stderr, "  \"%s\" %.1f%%", tok.decode({order[k]}).c_str(),
                         100.0 * std::exp((double)lg[order[k]] - lg[order[0]]) / z);
        std::fprintf(stderr, "\n\n");
        std::fflush(stderr);

        std::printf("%s", argv[2]);
        std::fflush(stdout);

        // Generation: one new token per call; the cache holds everything before it.
        std::vector<int32_t> out;
        std::string shown;
        auto g0 = Clock::now();
        for (int step = 0; step < n_gen; ++step) {
            int32_t next = (int32_t)(std::max_element(lg.begin(), lg.end()) - lg.begin());
            if (next == tok.eos()) break;
            out.push_back(next);
            std::string text = tok.decode(out);       // print only the new part; keeps multi-byte chars intact
            if (text.size() > shown.size()) { std::printf("%s", text.c_str() + shown.size()); std::fflush(stdout); shown = text; }
            if (step + 1 < n_gen) lg = model->eval({next}, n_threads);
        }
        const double gen_s = since(g0);
        std::printf("\n");
        std::fflush(stdout);
        std::fprintf(stderr, "\nprompt %.1f tok/s | generation: %zu tokens in %.1fs (%.2f tok/s)\n",
                     prompt.size() / prompt_s, out.size(), gen_s, out.size() / gen_s);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
