#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tierlm {

// GPT-2 style byte-level BPE, loaded from GGUF metadata.
class Tokenizer {
public:
    static Tokenizer from_gguf(const std::string& path);   // throws on failure / unsupported type

    // User-defined added tokens (e.g. whitespace runs) are always matched before BPE.
    // Control tokens like <|endoftext|> are matched only when parse_special is true.
    std::vector<int32_t> encode(const std::string& text, bool parse_special = false) const;
    std::string decode(const std::vector<int32_t>& ids) const;

    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }
    size_t vocab_size() const { return id_to_token_.size(); }

private:
    std::vector<std::string> id_to_token_;
    std::vector<int32_t> token_type_;                           // GGUF: 1 normal, 3 control, 4 user-defined, ...
    std::vector<std::pair<std::string, int32_t>> added_;        // added tokens, longest first
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;   // "left right" -> rank (lower merges first)
    std::string byte_to_uni_[256];
    std::unordered_map<std::string, uint8_t> uni_to_byte_;
    int32_t bos_ = -1, eos_ = -1;

    void bpe(const std::string& chunk, std::vector<int32_t>& out) const;
    void encode_plain(const std::string& text, std::vector<int32_t>& out) const;
};

}
