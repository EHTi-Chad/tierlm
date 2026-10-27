#include "tokenizer.h"
#include "unicode_tables.h"
#include "gguf.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace tierlm {
namespace {

template <size_t N>
bool in(const uni::Range (&t)[N], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < t[mid].lo) hi = mid;
        else if (cp > t[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}
bool is_letter(uint32_t c) { return in(uni::kLetter, c); }
bool is_number(uint32_t c) { return in(uni::kNumber, c); }
bool is_space(uint32_t c)  { return in(uni::kSpace, c); }
bool is_other(uint32_t c)  { return !is_letter(c) && !is_number(c) && !is_space(c); }

struct Cp { uint32_t cp; size_t off, len; };   // code point + its byte span in the input

std::vector<Cp> utf8_decode(const std::string& s) {
    std::vector<Cp> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        uint32_t cp; size_t n;
        if (c < 0x80)                                   { cp = c;        n = 1; }
        else if ((c >> 5) == 0x6  && i + 1 < s.size())  { cp = c & 0x1F; n = 2; }
        else if ((c >> 4) == 0xE  && i + 2 < s.size())  { cp = c & 0x0F; n = 3; }
        else if ((c >> 3) == 0x1E && i + 3 < s.size())  { cp = c & 0x07; n = 4; }
        else                                            { cp = 0xFFFD;   n = 1; }  // invalid byte: keep it as-is
        for (size_t k = 1; k < n; ++k) cp = (cp << 6) | (s[i + k] & 0x3F);
        out.push_back({cp, i, n});
        i += n;
    }
    return out;
}

std::string utf8_encode(uint32_t cp) {
    std::string s;
    if (cp < 0x80)         { s += char(cp); }
    else if (cp < 0x800)   { s += char(0xC0 | (cp >> 6));  s += char(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { s += char(0xE0 | (cp >> 12)); s += char(0x80 | ((cp >> 6) & 0x3F)); s += char(0x80 | (cp & 0x3F)); }
    else                   { s += char(0xF0 | (cp >> 18)); s += char(0x80 | ((cp >> 12) & 0x3F));
                             s += char(0x80 | ((cp >> 6) & 0x3F)); s += char(0x80 | (cp & 0x3F)); }
    return s;
}

// GPT-2 split pattern, hand-compiled:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
std::vector<std::pair<size_t, size_t>> pretokenize_gpt2(const std::string& s) {
    const auto cps = utf8_decode(s);
    const size_t n = cps.size();
    std::vector<std::pair<size_t, size_t>> out;
    auto C = [&](size_t k) { return cps[k].cp; };
    auto emit = [&](size_t a, size_t b) {           // code points [a, b) -> byte span
        size_t off = cps[a].off, end = cps[b - 1].off + cps[b - 1].len;
        out.push_back({off, end - off});
    };

    size_t i = 0;
    while (i < n) {
        if (C(i) == '\'' && i + 1 < n) {             // contractions
            uint32_t a = C(i + 1), b = i + 2 < n ? C(i + 2) : 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd')                                { emit(i, i + 2); i += 2; continue; }
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l'))  { emit(i, i + 3); i += 3; continue; }
        }
        size_t j = (C(i) == ' ' && i + 1 < n) ? i + 1 : i;   // optional single leading space
        uint32_t h = C(j);
        bool (*cls)(uint32_t) = is_letter(h) ? is_letter : is_number(h) ? is_number : is_other(h) ? is_other : nullptr;
        if (cls) {
            size_t k = j + 1;
            while (k < n && cls(C(k))) ++k;
            emit(i, k); i = k; continue;
        }
        size_t k = i;                                // whitespace run
        while (k < n && is_space(C(k))) ++k;
        if (k == n || k - i == 1) { emit(i, k); i = k; }           // \s+ to end, or a lone space char
        else                      { emit(i, k - 1); i = k - 1; }   // leave the last one to prefix the next word
    }
    return out;
}

// Llama 3 split pattern (Qwen2 is the same with single digits instead of groups of up to three), hand-compiled:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
std::vector<std::pair<size_t, size_t>> pretokenize_llama3(const std::string& s, int max_digits) {
    const auto cps = utf8_decode(s);
    const size_t n = cps.size();
    std::vector<std::pair<size_t, size_t>> out;
    auto C = [&](size_t k) { return cps[k].cp; };
    auto lower = [](uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; };
    auto nl = [](uint32_t c) { return c == '\r' || c == '\n'; };
    auto emit = [&](size_t a, size_t b) {
        size_t off = cps[a].off, end = cps[b - 1].off + cps[b - 1].len;
        out.push_back({off, end - off});
    };
    size_t i = 0;
    while (i < n) {
        // 1. contractions, any case
        if (C(i) == '\'' && i + 1 < n) {
            uint32_t a = lower(C(i + 1)), b = i + 2 < n ? lower(C(i + 2)) : 0;
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) { emit(i, i + 3); i += 3; continue; }
            if (a == 's' || a == 't' || a == 'm' || a == 'd')                                { emit(i, i + 2); i += 2; continue; }
        }
        // 2. letters, with at most one leading non-letter, non-digit, non-newline character
        {
            size_t j = i;
            if (!is_letter(C(i)) && !is_number(C(i)) && !nl(C(i)) && i + 1 < n && is_letter(C(i + 1))) j = i + 1;
            if (is_letter(C(j))) {
                size_t k = j + 1;
                while (k < n && is_letter(C(k))) ++k;
                emit(i, k); i = k; continue;
            }
        }
        // 3. digits, up to max_digits at a time
        if (is_number(C(i))) {
            size_t k = i + 1;
            while (k < n && k - i < (size_t)max_digits && is_number(C(k))) ++k;
            emit(i, k); i = k; continue;
        }
        // 4. optional space, a run of punctuation/symbols, then any newlines
        {
            size_t j = (C(i) == ' ' && i + 1 < n && is_other(C(i + 1))) ? i + 1 : i;
            if (is_other(C(j))) {
                size_t k = j + 1;
                while (k < n && is_other(C(k))) ++k;
                while (k < n && nl(C(k))) ++k;
                emit(i, k); i = k; continue;
            }
        }
        // 5-7. whitespace
        size_t k = i;
        while (k < n && is_space(C(k))) ++k;
        size_t last_nl = n;
        for (size_t m = i; m < k; ++m) if (nl(C(m))) last_nl = m;
        if (last_nl != n)          { emit(i, last_nl + 1); i = last_nl + 1; }   // \s*[\r\n]+
        else if (k == n)           { emit(i, k); i = k; }                       // \s+ at the end
        else if (k - i >= 2)       { emit(i, k - 1); i = k - 1; }               // \s+(?!\S): leave one for the next word
        else                       { emit(i, k); i = k; }                       // \s+
    }
    return out;
}

}  // namespace

Tokenizer Tokenizer::from_gguf(const std::string& path) {
    gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) throw std::runtime_error("failed to read GGUF: " + path);

    auto sval = [&](const char* k) -> std::string {
        int64_t id = gguf_find_key(g, k);
        return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING ? gguf_get_val_str(g, id) : "";
    };
    auto bval = [&](const char* k, bool def) -> bool {
        int64_t id = gguf_find_key(g, k);
        return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_BOOL ? gguf_get_val_bool(g, id) : def;
    };
    const std::string model = sval("tokenizer.ggml.model");
    Tokenizer t;
    if (model == "gpt2") {
        t.kind_ = Kind::Bpe;
        const std::string pre = sval("tokenizer.ggml.pre");
        if (pre == "llama-bpe" || pre == "llama3" || pre == "smaug-bpe" || pre == "dbrx") t.pre_ = Pre::Llama3;
        else if (pre == "qwen2" || pre == "deepseek-r1-qwen" || pre == "megrez") t.pre_ = Pre::Qwen2;
        else if (pre.empty() || pre == "default" || pre == "gpt-2" || pre == "olmo" || pre == "gpt2") t.pre_ = Pre::Gpt2;
        else {
            // Unknown splitter: most recent BPE models use the Llama 3 pattern or something close to it.
            std::fprintf(stderr, "warning: tokenizer pre-splitter '%s' is not known yet; using the Llama 3 one\n", pre.c_str());
            t.pre_ = Pre::Llama3;
        }
    } else if (model == "llama") {
        t.kind_ = Kind::Spm;                          // SentencePiece (Llama 2, Mistral 7B, Mixtral, ...)
        t.add_space_prefix_ = bval("tokenizer.ggml.add_space_prefix", true);
    } else {
        gguf_free(g);
        throw std::runtime_error("the '" + model + "' tokenizer is not supported yet");
    }
    int64_t tk = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (tk < 0) { gguf_free(g); throw std::runtime_error("GGUF has no tokenizer vocabulary"); }
    const size_t nt = gguf_get_arr_n(g, tk);
    t.id_to_token_.resize(nt);
    for (size_t i = 0; i < nt; ++i) {
        t.id_to_token_[i] = gguf_get_arr_str(g, tk, i);
        t.token_to_id_.emplace(t.id_to_token_[i], (int32_t)i);
    }
    int64_t yk = gguf_find_key(g, "tokenizer.ggml.token_type");
    t.token_type_.assign(nt, 1);
    if (yk >= 0 && gguf_get_arr_type(g, yk) == GGUF_TYPE_INT32 && gguf_get_arr_n(g, yk) == nt) {
        const int32_t* ty = (const int32_t*)gguf_get_arr_data(g, yk);
        t.token_type_.assign(ty, ty + nt);
    }
    for (size_t i = 0; i < nt; ++i)
        if ((t.token_type_[i] == 3 || t.token_type_[i] == 4) && !t.id_to_token_[i].empty())
            t.added_.push_back({t.id_to_token_[i], (int32_t)i});
    std::stable_sort(t.added_.begin(), t.added_.end(),
                     [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    int64_t sk = gguf_find_key(g, "tokenizer.ggml.scores");
    if (sk >= 0 && gguf_get_arr_type(g, sk) == GGUF_TYPE_FLOAT32 && gguf_get_arr_n(g, sk) == nt) {
        const float* sc = (const float*)gguf_get_arr_data(g, sk);
        t.score_.assign(sc, sc + nt);
    }
    if (t.kind_ == Kind::Spm && t.score_.empty()) { gguf_free(g); throw std::runtime_error("SentencePiece vocabulary has no scores"); }

    int64_t rk = gguf_find_key(g, "tokenizer.ggml.merges");          // optional: a byte-only vocab has none
    const size_t nm = rk < 0 ? 0 : gguf_get_arr_n(g, rk);
    for (size_t i = 0; i < nm; ++i) t.merge_rank_.emplace(gguf_get_arr_str(g, rk, i), (int32_t)i);

    auto u32 = [&](const char* k) -> int32_t {
        int64_t id = gguf_find_key(g, k);
        return id < 0 ? -1 : (int32_t)gguf_get_val_u32(g, id);
    };
    t.bos_ = u32("tokenizer.ggml.bos_token_id");
    t.eos_ = u32("tokenizer.ggml.eos_token_id");
    t.eot_ = u32("tokenizer.ggml.eot_token_id");
    t.unk_ = u32("tokenizer.ggml.unknown_token_id");
    t.chat_template_ = sval("tokenizer.chat_template");
    gguf_free(g);

    int extra = 0;                                   // GPT-2 bytes_to_unicode()
    for (int b = 0; b < 256; ++b) {
        bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        t.byte_to_uni_[b] = utf8_encode(direct ? (uint32_t)b : 256u + extra++);
        t.uni_to_byte_[t.byte_to_uni_[b]] = (uint8_t)b;
    }
    return t;
}

void Tokenizer::bpe(const std::string& chunk, std::vector<int32_t>& out) const {
    std::vector<std::string> sym;
    for (unsigned char c : chunk) sym.push_back(byte_to_uni_[c]);

    while (sym.size() > 1) {
        int best = INT_MAX; size_t at = 0;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            auto it = merge_rank_.find(sym[i] + " " + sym[i + 1]);
            if (it != merge_rank_.end() && it->second < best) { best = it->second; at = i; }
        }
        if (best == INT_MAX) break;
        const std::string a = sym[at], b = sym[at + 1];   // merge every occurrence of the winning pair
        std::vector<std::string> next;
        next.reserve(sym.size());
        for (size_t i = 0; i < sym.size();) {
            if (i + 1 < sym.size() && sym[i] == a && sym[i + 1] == b) { next.push_back(a + b); i += 2; }
            else { next.push_back(sym[i]); i += 1; }
        }
        sym.swap(next);
    }
    for (const auto& s : sym) {
        auto it = token_to_id_.find(s);
        if (it == token_to_id_.end()) throw std::runtime_error("BPE produced a token not in the vocab");
        out.push_back(it->second);
    }
}

void Tokenizer::encode_plain(const std::string& text, std::vector<int32_t>& out, bool first) const {
    if (kind_ == Kind::Spm) { spm(text, out, first); return; }
    const auto parts = pre_ == Pre::Gpt2 ? pretokenize_gpt2(text) : pretokenize_llama3(text, pre_ == Pre::Qwen2 ? 1 : 3);
    for (const auto& [off, len] : parts) bpe(text.substr(off, len), out);
}

// SentencePiece BPE: spaces become U+2581 ("▁"), then adjacent pieces are merged, highest-scoring merge
// first, as long as the merged piece is in the vocabulary. Pieces not in the vocabulary fall back to bytes.
void Tokenizer::spm(const std::string& raw, std::vector<int32_t>& out, bool prefix) const {
    std::string text = (prefix && add_space_prefix_) ? " " + raw : raw;
    std::string s;
    for (char ch : text) { if (ch == ' ') s += "\xE2\x96\x81"; else s += ch; }
    std::vector<std::string> sym;
    for (const auto& c : utf8_decode(s)) sym.push_back(s.substr(c.off, c.len));
    // score[i]: score of merging sym[i] and sym[i + 1] (-inf if that piece isn't in the vocabulary).
    // After a merge only the two neighbouring pairs change, so only they are looked up again.
    auto pair_score = [&](size_t i) -> float {
        auto it = token_to_id_.find(sym[i] + sym[i + 1]);
        return it == token_to_id_.end() ? -INFINITY : score_[it->second];
    };
    std::vector<float> score(sym.size() > 1 ? sym.size() - 1 : 0);
    for (size_t i = 0; i < score.size(); ++i) score[i] = pair_score(i);
    while (!score.empty()) {
        size_t at = 0;
        for (size_t i = 1; i < score.size(); ++i) if (score[i] > score[at]) at = i;   // highest, leftmost on ties
        if (score[at] == -INFINITY) break;
        sym[at] += sym[at + 1];
        sym.erase(sym.begin() + (long)at + 1);
        score.erase(score.begin() + (long)at);
        if (at < score.size()) score[at] = pair_score(at);
        if (at > 0) score[at - 1] = pair_score(at - 1);
    }
    for (const auto& p : sym) {
        auto it = token_to_id_.find(p);
        if (it != token_to_id_.end()) { out.push_back(it->second); continue; }
        for (unsigned char b : p) {                           // byte fallback: <0xAB>
            char name[8];
            std::snprintf(name, sizeof(name), "<0x%02X>", b);
            auto bt = token_to_id_.find(name);
            if (bt != token_to_id_.end()) out.push_back(bt->second);
            else if (unk_ >= 0) out.push_back(unk_);
        }
    }
}

// Added tokens are split out first (leftmost-longest), like Hugging Face does;
// only the text between them goes through the pre-tokenizer and BPE.
std::vector<int32_t> Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int32_t> ids;
    size_t start = 0, i = 0;
    bool prev_special = false;
    while (i < text.size()) {
        int32_t hit = -1; size_t hit_len = 0;
        for (const auto& [s, id] : added_) {
            if (!parse_special && token_type_[id] == 3) continue;
            if (text.compare(i, s.size(), s) == 0) { hit = id; hit_len = s.size(); break; }
        }
        if (hit < 0) { ++i; continue; }
        if (i > start) encode_plain(text.substr(start, i - start), ids, start == 0 || prev_special);
        ids.push_back(hit);
        prev_special = true;
        i += hit_len;
        start = i;
    }
    if (start < text.size()) encode_plain(text.substr(start), ids, start == 0 || prev_special);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || (size_t)id >= id_to_token_.size()) continue;
        const std::string& tok = id_to_token_[id];
        if (token_type_[id] == 3 || token_type_[id] == 4) { out += tok; continue; }   // added tokens are raw text
        if (kind_ == Kind::Spm) {
            if (token_type_[id] == 6 && tok.size() == 6 && tok.compare(0, 3, "<0x") == 0) {   // byte token <0xAB>
                out += (char)std::strtol(tok.substr(3, 2).c_str(), nullptr, 16);
                continue;
            }
            for (size_t k = 0; k < tok.size();) {
                if (tok.compare(k, 3, "\xE2\x96\x81") == 0) { out += ' '; k += 3; } else out += tok[k++];
            }
            continue;
        }
        for (const auto& c : utf8_decode(tok)) {
            std::string ch = tok.substr(c.off, c.len);
            auto it = uni_to_byte_.find(ch);
            out += it != uni_to_byte_.end() ? std::string(1, (char)it->second) : ch;
        }
    }
    // SentencePiece adds a space before the text; take it back off, like Hugging Face's decoder does.
    if (kind_ == Kind::Spm && add_space_prefix_ && !out.empty() && out[0] == ' ') out.erase(0, 1);
    return out;
}

}  // namespace tierlm
