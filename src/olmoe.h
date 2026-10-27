#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_sched;

namespace tierlm {

struct OlmoeHparams {
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, n_ff = 0;
    int n_expert = 0, n_expert_used = 0, n_vocab = 0, n_ctx = 0;
    float rope_base = 10000.0f, rms_eps = 1e-5f;
};

// OLMoE forward pass through ggml's backend API, so it runs on any ggml device
// (CPU, Vulkan, CUDA, ...). All weights and the KV cache live on one device for now;
// the tier loader will later spread them across devices.
class OlmoeModel {
public:
    // device: "auto" (the biggest dedicated GPU, else an integrated one, else the CPU), "cpu", "gpu",
    // or a specific ggml device name such as "Vulkan0" or "CUDA0".
    // max_ctx = 0 uses the model's own context length.
    static std::unique_ptr<OlmoeModel> load(const std::string& path, int max_ctx = 0,
                                            const std::string& device = "auto");   // throws on failure
    ~OlmoeModel();
    OlmoeModel(const OlmoeModel&) = delete;
    OlmoeModel& operator=(const OlmoeModel&) = delete;

    // Appends `ids` after the tokens already in the cache; returns logits for the last one.
    // n_threads applies to whatever runs on the CPU. If `routing` is given, it receives the experts
    // the router picked: n_tok x n_layer x n_expert_used ids, best first.
    std::vector<float> eval(const std::vector<int32_t>& ids, int n_threads, std::vector<int32_t>* routing = nullptr);
    void reset() { n_past_ = 0; }                 // forget the conversation
    void truncate(int n) { if (n < n_past_) n_past_ = n < 0 ? 0 : n; }   // keep only the first n tokens
    int n_past() const { return n_past_; }
    int max_ctx() const { return max_ctx_; }
    const std::string& device_name() const { return device_name_; }
    uint64_t weights_bytes() const { return weights_bytes_; }
    uint64_t kv_bytes() const { return kv_bytes_; }
    bool on_gpu() const { return main_ != cpu_; }
    bool integrated_gpu() const;                  // shares system memory (e.g. Intel Arc in a laptop)
    void device_memory(size_t& free, size_t& total) const;   // of the device holding the weights
    bool flash_attn() const { return flash_attn_; }
    const char* kv_type() const { return kv_f32_ ? "f32" : "f16"; }

    // Convenience: reset, then evaluate the whole sequence.
    std::vector<float> forward(const std::vector<int32_t>& ids, int n_threads) { reset(); return eval(ids, n_threads); }

    const OlmoeHparams& hparams() const { return hp_; }

private:
    OlmoeModel() = default;
    struct Graph {
        ggml_context* ctx = nullptr;
        struct ggml_cgraph* gf = nullptr;
        ggml_tensor *inp = nullptr, *pos = nullptr, *mask = nullptr, *logits = nullptr;
        std::vector<ggml_tensor*> orders;        // router rankings per layer (when routing is wanted)
    };
    Graph build(int n_tok, int n_past, bool want_routing);
    std::vector<float> eval_batch(const int32_t* ids, int n_tok, int n_threads, std::vector<int32_t>* routing);
    struct Layer {
        ggml_tensor *attn_norm, *wq, *wk, *wv, *wo, *q_norm, *k_norm;
        ggml_tensor *ffn_norm, *router, *up, *gate, *down;
        ggml_tensor *k_cache, *v_cache;          // [head_dim, n_head_kv, max_ctx]
    };
    OlmoeHparams hp_;
    ggml_backend* main_ = nullptr;               // device holding weights + cache
    ggml_backend* cpu_ = nullptr;                // fallback for ops the device can't run
    ggml_backend_sched* sched_ = nullptr;
    ggml_context* wctx_ = nullptr;               // weight tensor descriptors
    ggml_context* kvctx_ = nullptr;              // KV cache tensor descriptors
    ggml_backend_buffer* wbuf_ = nullptr;        // weight memory on the device
    ggml_backend_buffer* kvbuf_ = nullptr;       // KV cache memory on the device
    std::vector<uint8_t> graph_mem_;             // scratch for building graphs
    ggml_tensor *tok_embd_ = nullptr, *out_norm_ = nullptr, *output_ = nullptr;
    std::vector<Layer> layers_;
    std::string device_name_;
    int max_ctx_ = 0, n_past_ = 0;
    uint64_t weights_bytes_ = 0, kv_bytes_ = 0;
    void* dev_ = nullptr;                        // ggml_backend_dev_t of the main device
    bool flash_attn_ = true, kv_f32_ = false;
};

}
