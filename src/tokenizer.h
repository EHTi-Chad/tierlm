#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tierlm {

// Tokenizers loaded from GGUF metadata: byte-level BPE (GPT-2, Llama 3, Qwen, OLMoE, ...), SentencePiece
// (Llama 2, Mistral 7B, Mixtral, ...) and WordPiece (BERT-style embedding models: bge, MiniLM, nomic-embed-text).
class Tokenizer {
public:
    static Tokenizer from_gguf(const std::string& path);   // throws on failure / unsupported type

    // User-defined added tokens (e.g. whitespace runs) are always matched before BPE.
    // Control tokens like <|endoftext|> are matched only when parse_special is true.
    std::vector<int32_t> encode(const std::string& text, bool parse_special = false) const;
    // encode, from `from` on: a place just after an added token (or 0), whose tokens before it the caller has (text
    // before `from` isn't read). `marks`, if given, gets each added token's end in this
    // part: (the byte after it, how many of this part's tokens come before that byte). The tokens of a text are its
    // tokens up to any such mark followed by the tokens from that mark, so a text that only changes further on than
    // a mark plus added_max_len() bytes needn't be encoded again before it.
    std::vector<int32_t> encode_from(const std::string& text, bool parse_special, size_t from,
                                     std::vector<std::pair<size_t, size_t>>* marks = nullptr) const;
    size_t added_max_len() const { return added_max_; }
    // strip_prefix: SentencePiece puts a space before the text it encodes; drop it when decoding a whole reply
    // (true), keep it when the tokens continue earlier text (false).
    std::string decode(const std::vector<int32_t>& ids, bool strip_prefix = true) const;
    // A reply as it's written: decode(ids, strip) is the ids appended one by one to an empty string with
    // decode_append, then the first strip_len() bytes of it left out (with strip_prefix). Each token's text is
    // added once, instead of the whole reply decoded again for every token.
    void decode_append(std::string& out, int32_t id) const;
    size_t strip_len(const std::string& decoded) const {
        return kind_ == Kind::Spm && add_space_prefix_ && !decoded.empty() && decoded[0] == ' ' ? 1 : 0;
    }
    bool add_bos() const { return add_bos_; }                // whether raw prompts should start with bos()

    // Control tokens (like <|endoftext|> or <|user|>) mark structure, not text.
    bool is_control(int32_t id) const { return id >= 0 && (size_t)id < token_type_.size() && token_type_[id] == 3; }
    int32_t find(const std::string& token) const { auto it = token_to_id_.find(token); return it == token_to_id_.end() ? -1 : it->second; }

    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }
    int32_t eot() const { return eot_; }                      // end of turn, if the model has a separate one
    const std::string& chat_template() const { return chat_template_; }
    bool spm() const { return kind_ == Kind::Spm; }
    bool wordpiece() const { return kind_ == Kind::Wpm; }
    int32_t cls() const { return cls_; }                     // WordPiece: [CLS] goes first, [SEP] last
    int32_t sep() const { return sep_; }
    size_t vocab_size() const { return id_to_token_.size(); }

private:
    std::vector<std::string> id_to_token_;
    std::vector<int32_t> token_type_;                           // GGUF: 1 normal, 3 control, 4 user-defined, ...
    std::vector<std::pair<std::string, int32_t>> added_;        // added tokens, longest first
    bool added_first_[256] = {};                                // a byte some added token starts with
    size_t added_max_ = 0;                                      // the longest added token, in bytes
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;   // "left right" -> rank (lower merges first)
    std::string byte_to_uni_[256];
    std::unordered_map<std::string, uint8_t> uni_to_byte_;
    int32_t bos_ = -1, eos_ = -1, eot_ = -1, unk_ = -1, cls_ = -1, sep_ = -1;
    enum class Kind { Bpe, Spm, Wpm } kind_ = Kind::Bpe;
    bool lowercase_ = true;                                     // WordPiece: uncased vocabulary (lower case, no accents)
    enum class Pre { Gpt2, Llama3, Qwen2, Qwen35, O200k, Kimi, DeepseekV3, Tekken } pre_ = Pre::Gpt2;   // how BPE text is split before merging
    bool add_space_prefix_ = true, add_bos_ = false;                              // SentencePiece: " " before the text
    std::vector<float> score_;                                  // SentencePiece merge scores
    std::string chat_template_;

    void bpe(const std::string& chunk, std::vector<int32_t>& out) const;
    void spm(const std::string& text, std::vector<int32_t>& out, bool prefix) const;
    void wpm(const std::string& text, std::vector<int32_t>& out) const;
    void encode_plain(const std::string& text, std::vector<int32_t>& out, bool first) const;
};

}
