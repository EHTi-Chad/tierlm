#pragma once
// Qwen3.8 Flash-Next (qwen4exp): the hashed n-gram rows of its per-layer embedding (PLE), as Hugging Face's
// Qwen4ExpTextNGramEmbedding computes them. For each token, its 2-gram and 3-gram (the token with the one or two
// before it) are hashed into heads_per_ngram tables each; every table has its own prime number of rows (the first
// primes from ngram_vocab_size_base on), all stored one after another in one big embedding. A token before the
// start of the conversation, or before an end-of-text token, counts as that end-of-text token.
#include <cstdint>
#include <utility>
#include <vector>

namespace tierlm {

struct NgramHash {
    int ngram = 3, heads_per = 8;               // n-grams 2..ngram, heads_per tables each
    int32_t eos = 0;
    std::vector<int64_t> mult;                  // per position in the n-gram: an odd multiplier
    std::vector<int64_t> size, offset;          // per table: rows, first row in the embedding

    static uint64_t splitmix64(uint64_t v) {
        v += 0x9E3779B97F4A7C15ull;
        v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
        v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
        return v ^ (v >> 31);
    }
    static bool prime(int64_t v) {
        if (v < 2) return false;
        if (v % 2 == 0) return v == 2;
        for (int64_t d = 3; d * d <= v; d += 2) if (v % d == 0) return false;
        return true;
    }
    // ple_index: which PLE layer this is (0 for the first); vocab: the token vocabulary's size
    void init(int64_t vocab, int64_t base, int ple_index, int64_t seed, int ngram_, int heads_per_, int32_t eos_) {
        ngram = ngram_; heads_per = heads_per_; eos = eos_;
        const int heads = (ngram - 1) * heads_per;
        size.assign(heads, 0); offset.assign(heads, 0);
        // table g (counting all PLE layers' tables) has the (g+1)-th prime at or after base rows
        int64_t p = base - 1, total = 0;
        const int first = ple_index * heads;
        for (int g = 0; g < first + heads; ++g) { ++p; while (!prime(p)) ++p; if (g >= first) { size[g - first] = p; } }
        for (int h = 0; h < heads; ++h) { offset[h] = total; total += size[h]; }
        const uint64_t max_long = ((uint64_t)1 << 63) - 1;
        const uint64_t mmax = max_long / (uint64_t)(vocab > 1 ? vocab : 1), half = mmax / 2 > 0 ? mmax / 2 : 1;
        const uint64_t base_seed = (uint64_t)(seed + 10007 * (int64_t)ple_index);
        mult.assign(ngram, 0);
        for (int i = 0; i < ngram; ++i) mult[i] = (int64_t)(2 * (splitmix64(base_seed + 0x9E3779B97F4A7C15ull * (uint64_t)(i + 1)) % half) + 1);
    }
    // From the file's own numbers (qwen4exp.ple.*): the same as init() computes, without assuming the recipe.
    void set(int ngram_, int heads_per_, int32_t eos_, std::vector<int64_t> mult_, std::vector<int64_t> size_, std::vector<int64_t> offset_) {
        ngram = ngram_; heads_per = heads_per_; eos = eos_; mult = std::move(mult_); size = std::move(size_); offset = std::move(offset_);
    }
    int heads() const { return (ngram - 1) * heads_per; }
    // The rows for the token at position p of `tok` (the conversation so far): heads() ids into the embedding.
    void rows(const int32_t* tok, int64_t p, int32_t* out) const {
        std::vector<int64_t> t(ngram);
        bool cut = false;                        // an end-of-text token (or the start) between here and further back
        t[0] = tok[p];
        for (int s = 1; s < ngram; ++s) {
            const int64_t q = p - s;
            if (q < 0 || tok[q] == eos) cut = true;
            t[s] = cut ? eos : tok[q];
        }
        int k = 0;
        for (int n = 2; n <= ngram; ++n) {
            uint64_t mixed = (uint64_t)t[0] * (uint64_t)mult[0];          // wraps like torch's int64
            for (int s = 1; s < n; ++s) mixed ^= (uint64_t)t[s] * (uint64_t)mult[s];
            const int64_t x = (int64_t)mixed;
            for (int h = 0; h < heads_per; ++h, ++k) {
                int64_t r = x % size[k];
                if (r < 0) r += size[k];                                  // torch.remainder: the divisor's sign
                out[k] = (int32_t)(r + offset[k]);
            }
        }
    }
};

}  // namespace tierlm
