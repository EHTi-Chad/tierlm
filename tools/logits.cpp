// Prints the last-position logits for a list of token ids. Used by the tests.
#include "olmoe.h"
#include <cstdio>
#include <cstdlib>
#include <thread>

using namespace tierlm;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.gguf id [id ...]\n", argv[0]); return 1; }
    try {
        auto m = OlmoeModel::load(argv[1]);
        std::vector<int32_t> ids;
        for (int i = 2; i < argc; ++i) ids.push_back(std::atoi(argv[i]));
        auto lg = m->forward(ids, (int)std::thread::hardware_concurrency());
        for (float v : lg) std::printf("%.6g\n", v);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
