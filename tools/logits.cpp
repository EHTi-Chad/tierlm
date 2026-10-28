// Prints the last-position logits for a list of token ids. Used by the tests.
//   tierlm-logits model.gguf id [id ...]          whole sequence in one call
//   tierlm-logits model.gguf --step id [id ...]   first id alone, then one id per call (exercises the KV cache)
#include "model.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include "ggml-backend.h"
#ifdef GGML_USE_RPC
#include "ggml-rpc.h"
#endif

using namespace tierlm;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.gguf [--step] id [id ...]\n", argv[0]); return 1; }
    try {
        bool step = !std::strcmp(argv[2], "--step");
        std::vector<int32_t> ids;
        for (int i = step ? 3 : 2; i < argc; ++i) ids.push_back(std::atoi(argv[i]));
        // Test hooks: TIERLM_RPC=host:port adds an RPC device (seen as a GPU); TIERLM_VRAM_MB limits the VRAM
        // the tier loader may use, forcing a model to be split between the device and RAM.
#ifdef GGML_USE_RPC
        if (const char* ep = std::getenv("TIERLM_RPC")) ggml_backend_register(ggml_backend_rpc_add_server(ep));
#endif
        LoadOptions o;
        o.max_ctx = std::max<int>(512, (int)ids.size() + 8);
        if (const char* v = std::getenv("TIERLM_VRAM_MB")) o.vram_budget_mb = std::atoi(v);
        if (const char* v = std::getenv("TIERLM_SPLIT")) o.split_mode = v;
        auto m = Model::load(argv[1], o);
        if (std::getenv("TIERLM_SHOW_PLACEMENT")) {
            const auto& p = m->placement();
            std::fprintf(stderr, "%s: %s, VRAM %.2f MiB, RAM %.2f MiB, KV %d/%d, attention %d/%d, ffn %d/%d layers in VRAM\n",
                         m->device_name().c_str(), p.mode.c_str(), p.gpu_weights / 1048576.0, p.cpu_weights / 1048576.0,
                         p.kv_gpu_layers, p.layers, p.attn_gpu, p.layers, p.ffn_gpu, p.layers);
        }
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
