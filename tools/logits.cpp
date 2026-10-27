// Prints the last-position logits for a list of token ids. Used by the tests.
//   tierlm-logits model.gguf id [id ...]          whole sequence in one call
//   tierlm-logits model.gguf --step id [id ...]   first id alone, then one id per call (exercises the KV cache)
#include "olmoe.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace tierlm;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.gguf [--step] id [id ...]\n", argv[0]); return 1; }
    try {
        bool step = !std::strcmp(argv[2], "--step");
        std::vector<int32_t> ids;
        for (int i = step ? 3 : 2; i < argc; ++i) ids.push_back(std::atoi(argv[i]));
        auto m = OlmoeModel::load(argv[1], std::max<int>(512, (int)ids.size() + 8));
        const int nt = std::getenv("TIERLM_THREADS") ? std::atoi(std::getenv("TIERLM_THREADS")) : std::max(1, (int)std::thread::hardware_concurrency() / 2);

        std::vector<float> lg;
        if (!step) lg = m->eval(ids, nt);
        else for (int32_t id : ids) lg = m->eval({id}, nt);
        for (float v : lg) std::printf("%.6g\n", v);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
