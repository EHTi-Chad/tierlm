#pragma once
// Embedding models: encoders that read a whole text at once and turn it into one vector ("embedding"), used for
// search, document comparison and retrieval (RAG). Supported: BERT-style models (`bert`: bge, all-MiniLM, e5...)
// and Nomic's (`nomic-bert`: nomic-embed-text v1 / v1.5).
//
// They are small (about 0.1-0.6 GB), so the whole model lives on one device: the GPU when it has room, else the
// CPU. Unlike a chat model there is no KV cache and no generation: every token attends to every other (both
// directions), the last layer's hidden states are pooled (averaged, or the first token's) into one vector.
#include "model.h"
#include <memory>
#include <string>
#include <vector>

struct ggml_backend;
struct ggml_backend_sched;
struct ggml_backend_buffer;
struct ggml_context;
struct ggml_tensor;
struct gguf_context;

namespace tierlm {

bool is_encoder_arch(const std::string& arch);

class Encoder {
public:
    enum class Pooling { None = 0, Mean = 1, Cls = 2, Last = 3 };
    // device: "auto" (a GPU if it has room for the model with plenty to spare, else the CPU), "cpu" or a ggml
    // device name. Throws on failure.
    static std::unique_ptr<Encoder> load(const std::string& path, const std::string& device = "auto");
    ~Encoder();

    // The embedding of one text, given as token ids (with the model's [CLS]/[SEP] already around it), pooled as
    // the model says and not normalised. At most max_tokens() ids.
    std::vector<float> embed(const std::vector<int32_t>& ids, int n_threads);

    const std::string& arch() const { return arch_; }
    const std::string& device_name() const { return device_name_; }
    int n_embd() const { return n_embd_; }
    int max_tokens() const { return n_ctx_; }
    Pooling pooling() const { return pooling_; }
    size_t weight_bytes() const { return weight_bytes_; }
    bool on_gpu() const { return main_ != cpu_; }

private:
    Encoder() = default;
    struct Layer {
        ggml_tensor *qkv = nullptr, *qkv_b = nullptr, *q = nullptr, *q_b = nullptr, *k = nullptr, *k_b = nullptr,
                    *v = nullptr, *v_b = nullptr, *o = nullptr, *o_b = nullptr;
        ggml_tensor *attn_norm = nullptr, *attn_norm_b = nullptr;       // after attention (post-norm)
        ggml_tensor *up = nullptr, *up_b = nullptr, *gate = nullptr, *down = nullptr, *down_b = nullptr;
        ggml_tensor *out_norm = nullptr, *out_norm_b = nullptr;         // after the feed-forward block
    };
    std::string arch_, device_name_;
    int n_embd_ = 0, n_head_ = 0, n_layer_ = 0, n_ctx_ = 512;
    float eps_ = 1e-12f, rope_base_ = 10000.0f;
    bool rope_ = false;                       // nomic-bert: rotary positions; bert: learned position embeddings
    Pooling pooling_ = Pooling::Mean;
    ggml_tensor *tok_embd_ = nullptr, *type_embd_ = nullptr, *pos_embd_ = nullptr, *embd_norm_ = nullptr, *embd_norm_b_ = nullptr;
    std::vector<Layer> layers_;
    ggml_context* wctx_ = nullptr;
    gguf_context* gguf_ = nullptr;
    ggml_backend* cpu_ = nullptr;
    ggml_backend* main_ = nullptr;
    ggml_backend_buffer* wbuf_ = nullptr;
    ggml_backend_sched* sched_ = nullptr;
    size_t weight_bytes_ = 0;
};

}
