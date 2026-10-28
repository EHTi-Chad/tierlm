#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tierlm {

// Tokenizers loaded from GGUF metadata: byte-level BPE (GPT-2, Llama 3, Qwen, OLMoE, ...) and
// SentencePiece (Llama 2, Mistral 7B, Mixtral, ...).
class Tokenizer {
public:
    static Tokenizer from_gguf(const std::string& path);   // throws on failure / unsupported type

    // User-defined added tokens (e.g. whitespace runs) are always matched before BPE.
    // Control tokens like <|endoftext|> are matched only when parse_special is true.
    std::vector<int32_t> encode(const std::string& text, bool parse_special = false) const;
    // strip_prefix: SentencePiece puts a space before the text it encodes; drop it when decoding a whole reply
    // (true), keep it when the tokens continue earlier text (false).
    std::string decode(const std::vector<int32_t>& ids, bool strip_prefix = true) const;
    bool add_bos() const { return add_bos_; }                // whether raw prompts should start with bos()

    // Control tokens (like <|endoftext|> or <|user|>) mark structure, not text.
    bool is_control(int32_t id) const { return id >= 0 && (size_t)id < token_type_.size() && token_type_[id] == 3; }
    int32_t find(const std::string& token) const { auto it = token_to_id_.find(token); return it == token_to_id_.end() ? -1 : it->second; }

    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }
    int32_t eot() const { return eot_; }                      // end of turn, if the model has a separate one
    const std::string& chat_template() const { return chat_template_; }
    bool spm() const { return kind_ == Kind::Spm; }
    size_t vocab_size() const { return id_to_token_.size(); }

private:
    std::vector<std::string> id_to_token_;
    std::vector<int32_t> token_type_;                           // GGUF: 1 normal, 3 control, 4 user-defined, ...
    std::vector<std::pair<std::string, int32_t>> added_;        // added tokens, longest first
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;   // "left right" -> rank (lower merges first)
    std::string byte_to_uni_[256];
    std::unordered_map<std::string, uint8_t> uni_to_byte_;
    int32_t bos_ = -1, eos_ = -1, eot_ = -1, unk_ = -1;
    enum class Kind { Bpe, Spm } kind_ = Kind::Bpe;
    enum class Pre { Gpt2, Llama3, Qwen2, Qwen35 } pre_ = Pre::Gpt2;   // how BPE text is split before merging
    bool add_space_prefix_ = true, add_bos_ = false;                              // SentencePiece: " " before the text
    std::vector<float> score_;                                  // SentencePiece merge scores
    std::string chat_template_;

    void bpe(const std::string& chunk, std::vector<int32_t>& out) const;
    void spm(const std::string& text, std::vector<int32_t>& out, bool prefix) const;
    void encode_plain(const std::string& text, std::vector<int32_t>& out, bool first) const;
};

}
