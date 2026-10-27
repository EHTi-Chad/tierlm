#include "tokenizer.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using namespace tierlm;

static std::string unhex(const std::string& h) {
    std::string s;
    for (size_t i = 0; i + 1 < h.size(); i += 2) s += (char)std::stoi(h.substr(i, 2), nullptr, 16);
    return s;
}
static std::string show(const std::vector<int32_t>& v) {
    std::string s;
    for (auto x : v) s += std::to_string(x) + " ";
    return s;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.gguf cases.txt\n", argv[0]); return 1; }
    try {
        Tokenizer tok = Tokenizer::from_gguf(argv[1]);
        std::ifstream in(argv[2]);
        if (!in) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
        std::string line;
        int pass = 0, fail = 0;
        while (std::getline(in, line)) {
            size_t tab = line.find('\t');
            if (tab == std::string::npos) continue;
            std::string text = unhex(line.substr(0, tab));
            std::vector<int32_t> want;
            std::istringstream ss(line.substr(tab + 1));
            for (int32_t x; ss >> x;) want.push_back(x);

            auto got = tok.encode(text);
            bool round = tok.decode(got) == text;
            if (got == want && round) { ++pass; continue; }
            ++fail;
            std::printf("FAIL \"%s\"\n  want: %s\n  got:  %s\n  round-trip: %s\n",
                        text.c_str(), show(want).c_str(), show(got).c_str(), round ? "ok" : "BROKEN");
        }
        std::printf("%d passed, %d failed (vocab %zu)\n", pass, fail, tok.vocab_size());
        return fail ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
