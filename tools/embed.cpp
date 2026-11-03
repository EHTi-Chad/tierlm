// tierlm-embed: the embedding of a text (or of given token ids) from an embedding model, for tests and scripts.
//   tierlm-embed model.gguf "some text"          prints the token ids, then the embedding (pooled, not normalised)
//   tierlm-embed model.gguf --ids 101,2023,102   embeds exactly these ids
//   options: -d cpu|auto|<device>   -t <threads>   --rpc host:port (testing: a tierlm-rpc-server as a stand-in GPU)
#include "encoder.h"
#include "tokenizer.h"
#include "backends.h"
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>

using namespace tierlm;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s model.gguf (\"text\" | --ids 1,2,3) [-d cpu|auto|<device>] [-t threads]\n", argv[0]);
        return 1;
    }
    std::string path = argv[1], text, ids_arg, device = "cpu";
    int threads = (int)std::max(1u, std::thread::hardware_concurrency() / 2);
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--ids" && i + 1 < argc) ids_arg = argv[++i];
        else if (a == "-d" && i + 1 < argc) device = argv[++i];
        else if (a == "-t" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--rpc" && i + 1 < argc) { if (!add_rpc_server(argv[++i])) { std::fprintf(stderr, "error: RPC isn't built in\n"); return 1; } }
        else text = a;
    }
    try {
        auto enc = Encoder::load(path, device);
        std::vector<int32_t> ids;
        if (!ids_arg.empty()) {
            std::stringstream ss(ids_arg);
            for (std::string x; std::getline(ss, x, ',');) ids.push_back(std::atoi(x.c_str()));
        } else {
            Tokenizer tok = Tokenizer::from_gguf(path);
            if (tok.cls() >= 0) ids.push_back(tok.cls());
            for (int32_t t : tok.encode(text)) ids.push_back(t);
            if (tok.sep() >= 0) ids.push_back(tok.sep());
        }
        std::printf("tokens:");
        for (int32_t t : ids) std::printf(" %d", t);
        std::printf("\nembedding:");
        for (float v : enc->embed(ids, threads)) std::printf(" %.9g", v);
        std::printf("\n");
        std::fprintf(stderr, "%s on %s: %d numbers, %d tokens at most\n", enc->arch().c_str(), enc->device_name().c_str(),
                     enc->n_embd(), enc->max_tokens());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
