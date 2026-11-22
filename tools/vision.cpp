// Image reading, for the tests and for trying a vision file by hand.
//   tierlm-vision mmproj.gguf image.png out.f32 [max_tokens]
//       the image's image words (prints "rows cols dim"; out.f32: rows*cols*dim floats, row by row)
//   tierlm-vision mmproj.gguf --patches image.png out.f32 [max_tokens]
//       just the prepared patches (prints "gh gw"; out.f32: gh*gw patches of 3*16*16 numbers)
//   tierlm-vision mmproj.gguf --chat model.gguf image.png[,image2.png...] out.f32 MODE id ... -1 ... id
//       a chat model reading the image(s): each -1 in the ids marks where the next image's words go. Writes the last position's
//       logits to out.f32. MODE: "all" (one call), "step" (one id per call) or a number (calls of that many ids,
//       so an image can be cut in the middle). With the model's MTP draft block: "mtp" its prediction after the
//       last id, run over every place at once, "mtp-step" one place at a time, "mtp-N" N at a time; "gen" the 24
//       tokens the model writes next (greedy; out.f32: their ids), "gen-spec" the same guessing ahead.
// Environment: TIERLM_DEVICE (cpu, CUDA0, ...) as for the other tools; TIERLM_VISION_EXACT=1 plain attention.
#include "model.h"
#include "speculate.h"
#include "vision.h"
#include "backends.h"
#include <algorithm>
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
            LoadOptions o;
            if (dev) o.device = dev;
            o.max_ctx = 2048;
            const std::string how = argv[6];
            o.mtp = how.compare(0, 3, "mtp") == 0 || how == "gen-spec";
            auto m = Model::load(argv[3], o);
            struct Img { int32_t first; int n, rows, cols; };
            std::vector<Img> imgs;                            // (several images: their files with commas between)
            std::string list = argv[4];
            for (size_t a = 0; a <= list.size();) {
                size_t b = list.find(',', a);
                if (b == std::string::npos) b = list.size();
                const auto bytes = read_file(list.substr(a, b - a).c_str());
                const auto p = v->prepare(decode_image(bytes.data(), bytes.size()));
                const std::vector<float> words = v->encode(p, threads);
                const int rows = p.gh / v->merge(), ncol = p.gw / v->merge();
                imgs.push_back({m->add_image(0x1234 + (uint64_t)imgs.size(), words, rows, ncol), p.tokens(), rows, ncol});
                a = b + 1;
            }
            std::vector<int32_t> ids;
            size_t next = 0;
            for (int i = 7; i < argc; ++i) {
                const int id = std::atoi(argv[i]);
                if (id == -1) {
                    if (next >= imgs.size()) throw std::runtime_error("more -1 than images");
                    for (int k = 0; k < imgs[next].n; ++k) ids.push_back(imgs[next].first + k);
                    ++next;
                } else ids.push_back(id);
            }
            if (how == "gen" || how == "gen-spec") {              // what the model writes next, with and without guessing
                const int V = m->hparams().n_vocab, n_gen = 24;
                auto argmax = [&](const float* v) { return (int32_t)(std::max_element(v, v + V) - v); };
                std::vector<int32_t> out;
                if (how == "gen") {
                    std::vector<float> lg = m->eval(ids, threads);
                    while ((int)out.size() < n_gen) { out.push_back(argmax(lg.data())); lg = m->eval({out.back()}, threads); }
                } else {
                    Speculator sp(*m, std::getenv("TIERLM_DRAFT_N") ? std::atoi(std::getenv("TIERLM_DRAFT_N")) : 3);
                    std::vector<float> lg = sp.prompt(ids, threads);
                    int32_t x = argmax(lg.data());
                    out.push_back(x);
                    while ((int)out.size() < n_gen) {
                        auto prod = sp.step(x, threads, [&](const float* row, const std::vector<int32_t>&) { return argmax(row); }, [](int32_t) { return false; });
                        for (int32_t t : prod) out.push_back(t);
                        x = prod.back();
                    }
                    out.resize(n_gen);
                    std::printf("steps %d, drafted %d, accepted %d\n", sp.stats().steps, sp.stats().drafted, sp.stats().accepted);
                }
                write_floats(argv[5], std::vector<float>(out.begin(), out.end()));
                return 0;
            }
            if (how.compare(0, 3, "mtp") == 0) {                  // the draft block over the conversation, images and all
                const int E = m->hparams().hidden_width(), n = (int)ids.size();
                const int chunk = how == "mtp" ? n : how == "mtp-step" ? 1 : std::max(1, std::atoi(how.c_str() + 4));
                Model::EvalOut r = m->eval_ex(ids.data(), n, threads, 1, true, 1, nullptr);
                std::vector<float> h((size_t)n * E, 0.0f), logits;    // place t: the token at t beside the model's state at t - 1
                std::copy(r.hidden.begin(), r.hidden.begin() + (size_t)(n - 1) * E, h.begin() + E);
                for (int off = 0; off < n; off += chunk) {
                    const int k = std::min(chunk, n - off);
                    logits = m->mtp_eval(ids.data() + off, h.data() + (size_t)off * E, k, off, threads);
                }
                write_floats(argv[5], logits);
                return 0;
            }
            const int chunk = how == "all" ? (int)ids.size() : how == "step" ? 1 : std::max(1, std::atoi(how.c_str()));
            std::vector<float> logits;
            for (size_t off = 0; off < ids.size(); off += chunk) {
                std::vector<int32_t> part(ids.begin() + off, ids.begin() + std::min(ids.size(), off + chunk));
                logits = m->eval(part, threads);
            }
            write_floats(argv[5], logits);
            std::printf("%zu tokens, %zu image(s), the first %d image words (%dx%d)\n", ids.size(), imgs.size(), imgs[0].n, imgs[0].rows, imgs[0].cols);
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
