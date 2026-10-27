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

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.gguf \"prompt\" [-n tokens] [-t threads]\n", argv[0]); return 1; }
    int n_gen = 32, n_threads = (int)std::thread::hardware_concurrency();
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "-n")) n_gen = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "-t")) n_threads = std::atoi(argv[i + 1]);
    }
    try {
        auto t0 = std::chrono::steady_clock::now();
        Tokenizer tok = Tokenizer::from_gguf(argv[1]);
        auto model = OlmoeModel::load(argv[1]);
        auto t1 = std::chrono::steady_clock::now();
        const auto& hp = model->hparams();
        std::fprintf(stderr, "loaded in %.1fs | %d layers, %d experts (%d active), vocab %d, %d threads\n",
                     std::chrono::duration<double>(t1 - t0).count(), hp.n_layer, hp.n_expert, hp.n_expert_used,
                     hp.n_vocab, n_threads);

        std::vector<int32_t> ids = tok.encode(argv[2]);
        std::fprintf(stderr, "prompt: %zu tokens\n\n", ids.size());
        std::printf("%s", argv[2]);
        std::fflush(stdout);

        std::string shown;
        const size_t n_prompt = ids.size();
        auto g0 = std::chrono::steady_clock::now();
        for (int step = 0; step < n_gen; ++step) {
            std::vector<float> lg = model->forward(ids, n_threads);
            int32_t next = (int32_t)(std::max_element(lg.begin(), lg.end()) - lg.begin());

            if (step == 0) {   // show what the model was considering for the very first token
                float mx = lg[next];
                double z = 0;
                for (float v : lg) z += std::exp((double)v - mx);
                std::vector<int32_t> order(lg.size());
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                                  [&](int32_t a, int32_t b) { return lg[a] > lg[b]; });
                std::fprintf(stderr, "top 5 next tokens:");
                for (int k = 0; k < 5; ++k)
                    std::fprintf(stderr, "  \"%s\" %.1f%%", tok.decode({order[k]}).c_str(),
                                 100.0 * std::exp((double)lg[order[k]] - mx) / z);
                std::fprintf(stderr, "\n\n");
            }
            if (next == tok.eos()) break;
            ids.push_back(next);

            // Decode everything generated so far and print only the new part (keeps multi-byte chars intact).
            std::string text = tok.decode(std::vector<int32_t>(ids.begin() + n_prompt, ids.end()));
            if (text.size() > shown.size()) { std::printf("%s", text.c_str() + shown.size()); std::fflush(stdout); shown = text; }
        }
        auto g1 = std::chrono::steady_clock::now();
        size_t n_new = ids.size() - n_prompt;
        double secs = std::chrono::duration<double>(g1 - g0).count();
        std::fprintf(stderr, "\n\n%zu tokens in %.1fs (%.2f tok/s, no KV cache yet)\n", n_new, secs, n_new / secs);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
