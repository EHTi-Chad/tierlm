// tierlm-run: tokenize a prompt, run the model, and generate text greedily.
#include "model.h"
#include "tokenizer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>

using namespace tierlm;
using Clock = std::chrono::steady_clock;
static double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s model.gguf \"prompt\" [-n tokens] [-t threads] [-c context] [-d auto|cpu|gpu|<device>]\n"
                             "       [--split layers|experts] [--vram MiB]\n", argv[0]);
        return 1;
    }
    int n_gen = 32, n_threads = std::max(1, (int)std::thread::hardware_concurrency() / 2), n_ctx = 2048;
    std::string device = "auto", split = "layers";
    int vram_mb = 0;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "-n")) n_gen = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "-t")) n_threads = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "-c")) n_ctx = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "-d")) device = argv[i + 1];
        else if (!std::strcmp(argv[i], "--split")) split = argv[i + 1];
        else if (!std::strcmp(argv[i], "--vram")) vram_mb = std::atoi(argv[i + 1]);
    }
    try {
        auto t0 = Clock::now();
        Tokenizer tok = Tokenizer::from_gguf(argv[1]);
        LoadOptions o;
        o.max_ctx = n_ctx; o.device = device; o.split_mode = split; o.vram_budget_mb = vram_mb;
        auto model = Model::load(argv[1], o);
        const auto& hp = model->hparams();
        std::fprintf(stderr, "device: %s\nloaded in %.1fs | %d layers, %d experts (%d active), vocab %d, ctx %d, %d CPU threads\n",
                     model->device_name().c_str(), since(t0), hp.n_layer, hp.n_expert, hp.n_expert_used, hp.n_vocab, model->max_ctx(), n_threads);
        const auto& pl = model->placement();
        if (model->on_gpu())
            std::fprintf(stderr, "placement (%s split): %.2f GiB in VRAM, %.2f GiB in RAM | %d of %d layers fully in VRAM, KV cache %d/%d layers in VRAM\n",
                         pl.mode.c_str(), pl.gpu_weights / 1073741824.0, pl.cpu_weights / 1073741824.0,
                         pl.ffn_gpu, pl.layers, pl.kv_gpu_layers, pl.layers);

        // Prompt processing: the whole prompt in one call fills the KV cache.
        std::vector<int32_t> prompt = tok.encode(argv[2]);
        if (tok.add_bos()) prompt.insert(prompt.begin(), tok.bos());   // Llama-family models expect a start marker
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
            std::fprintf(stderr, "  \"%s\" %.1f%%", tok.decode({order[k]}, false).c_str(),
                         100.0 * std::exp((double)lg[order[k]] - lg[order[0]]) / z);
        std::fprintf(stderr, "\n\n");
        std::fflush(stderr);

        std::printf("%s", argv[2]);
        std::fflush(stdout);

        // Generation: one new token per call; the cache holds everything before it.
        std::vector<int32_t> out;
        std::string shown;
        model->reset_stats();
        auto g0 = Clock::now();
        for (int step = 0; step < n_gen; ++step) {
            int32_t next = (int32_t)(std::max_element(lg.begin(), lg.end()) - lg.begin());
            if (next == tok.eos() || next == tok.eot()) break;
            out.push_back(next);
            std::string text = tok.decode(out, false);       // print only the new part; keeps multi-byte chars intact
            if (text.size() > shown.size()) { std::printf("%s", text.c_str() + shown.size()); std::fflush(stdout); shown = text; }
            if (step + 1 < n_gen) lg = model->eval({next}, n_threads);
        }
        const double gen_s = since(g0);
        std::printf("\n");
        std::fflush(stdout);
        std::fprintf(stderr, "\nprompt %.1f tok/s | generation: %zu tokens in %.1fs (%.2f tok/s)\n",
                     prompt.size() / prompt_s, out.size(), gen_s, out.size() / gen_s);
        if (std::getenv("TIERLM_PROFILE")) {
            const auto& s = model->stats();
            const double n = std::max(1, s.calls);
            std::fprintf(stderr, "per token: build %.2f ms, alloc %.2f ms, inputs %.2f ms, compute %.2f ms, read logits %.2f ms"
                                 " | graph splits %d, copies %d | threads %d\n",
                         s.build_ms / n, s.alloc_ms / n, s.input_ms / n, s.compute_ms / n, s.output_ms / n, s.splits, s.copies, n_threads);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
