// Image reading, for the tests and for trying a vision file by hand.
//   tierlm-vision mmproj.gguf image.png out.f32 [max_tokens]
//       the image's image words (prints "rows cols dim"; out.f32: rows*cols*dim floats, row by row)
//   tierlm-vision mmproj.gguf --patches image.png out.f32 [max_tokens]
//       just the prepared patches (prints "gh gw"; out.f32: gh*gw patches of 3*16*16 numbers)
//   tierlm-vision mmproj.gguf --chat model.gguf image.png out.f32 MODE id ... -1 ... id
//       a chat model reading the image: -1 in the ids marks where its words go. Writes the last position's
//       logits to out.f32. MODE: "all" (one call), "step" (one id per call) or a number (calls of that many ids,
//       so an image can be cut in the middle).
// Environment: TIERLM_DEVICE (cpu, CUDA0, ...) as for the other tools; TIERLM_VISION_EXACT=1 plain attention.
#include "model.h"
#include "vision.h"
#include "backends.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace tierlm;

static std::vector<uint8_t> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("can't open ") + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static void write_floats(const char* path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char*)v.data(), (std::streamsize)(v.size() * sizeof(float)));
    if (!f) throw std::runtime_error(std::string("can't write ") + path);
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s mmproj.gguf image.png out.f32 [max_tokens]  (see the source for more)\n", argv[0]); return 1; }
    try {
        load_backends();
        if (const char* ep = std::getenv("TIERLM_RPC"))           // tests: a stand-in GPU (ggml's RPC backend)
            if (!add_rpc_server(ep)) throw std::runtime_error("TIERLM_RPC needs a build with -DGGML_RPC=ON");
        const char* dev = std::getenv("TIERLM_DEVICE");
        const int threads = std::max(1u, std::thread::hardware_concurrency() / 2);
        const std::string mode = argv[2];
        if (mode == "--patches") {
            auto v = Vision::load(argv[1], dev ? dev : "cpu");
            const auto bytes = read_file(argv[3]);
            const auto p = v->prepare(decode_image(bytes.data(), bytes.size()), argc > 5 ? std::atoi(argv[5]) : 0);
            write_floats(argv[4], p.patches);
            std::printf("%d %d\n", p.gh, p.gw);
            return 0;
        }
        if (mode == "--chat") {
            if (argc < 8) throw std::runtime_error("--chat needs model.gguf image.png out.f32 MODE ids");
            auto v = Vision::load(argv[1], dev ? dev : "cpu");
            const auto bytes = read_file(argv[4]);
            const auto p = v->prepare(decode_image(bytes.data(), bytes.size()));
            const std::vector<float> words = v->encode(p, threads);
            LoadOptions o;
            if (dev) o.device = dev;
            o.max_ctx = 2048;
            auto m = Model::load(argv[3], o);
            const int32_t first = m->add_image(0x1234, words, p.gh / v->merge(), p.gw / v->merge());
            std::vector<int32_t> ids;
            for (int i = 7; i < argc; ++i) {
                const int id = std::atoi(argv[i]);
                if (id == -1) for (int k = 0; k < p.tokens(); ++k) ids.push_back(first + k);
                else ids.push_back(id);
            }
            const std::string how = argv[6];
            const int chunk = how == "all" ? (int)ids.size() : how == "step" ? 1 : std::max(1, std::atoi(how.c_str()));
            std::vector<float> logits;
            for (size_t off = 0; off < ids.size(); off += chunk) {
                std::vector<int32_t> part(ids.begin() + off, ids.begin() + std::min(ids.size(), off + chunk));
                logits = m->eval(part, threads);
            }
            write_floats(argv[5], logits);
            std::printf("%zu tokens, %d image words (%dx%d)\n", ids.size(), p.tokens(), p.gh / v->merge(), p.gw / v->merge());
            return 0;
        }
        auto v = Vision::load(argv[1], dev ? dev : "cpu");
        const auto bytes = read_file(argv[2]);
        const auto p = v->prepare(decode_image(bytes.data(), bytes.size()), argc > 4 ? std::atoi(argv[4]) : 0);
        const std::vector<float> words = v->encode(p, threads);
        write_floats(argv[3], words);
        std::printf("%d %d %d\n", p.gh / v->merge(), p.gw / v->merge(), v->out_dim());
        std::fprintf(stderr, "vision on %s%s\n", v->device_name().c_str(), v->gpu_helps() ? (v->last_helped() ? " (the GPU helped)" : " (no room on the GPU)") : "");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
