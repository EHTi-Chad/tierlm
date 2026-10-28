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

// The architectures tierlm can run. Most modern models are variations on one transformer design; these
// switches describe the variations, so one graph builder covers them all.
//   llama     Llama 2/3, Mistral, Mixtral (MoE), and most Llama-style models
//   qwen2     Qwen2 / Qwen2.5: Llama-style, with biases on Q/K/V
//   qwen3     Qwen3 dense: per-head QK-norm, head size not tied to the model width
//   qwen3moe  Qwen3 MoE (e.g. Qwen3-30B-A3B)
//   olmoe     OLMoE: QK-norm over the full width, expert weights not renormalised
bool arch_supported(const std::string& arch);
std::vector<std::string> supported_archs();

struct Hparams {
    std::string arch;
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
    int n_ff = 0;                     // dense FFN width
    int n_ff_exp = 0;                 // width of each expert (MoE)
    int n_expert = 0, n_expert_used = 0, n_vocab = 0, n_ctx = 0;
    float rope_base = 10000.0f, rope_scale = 1.0f, rms_eps = 1e-5f;
    // Variations
    bool rope_neox = false;           // rotate halves (NeoX) instead of adjacent pairs (original Llama)
    enum QkNorm { kNone, kFull, kPerHead } qk_norm = kNone;
    bool qkv_bias = false;            // biases on Q/K/V projections
    bool expert_renorm = false;       // renormalise the chosen experts' weights to sum to 1
    bool expert_softmax_first = true; // softmax over all experts, then pick (vs pick then softmax: same order)
    bool is_moe() const { return n_expert > 0; }
};
using OlmoeHparams = Hparams;

// Load-time options. -1 / 0 mean "use the default".
struct LoadOptions {
    int max_ctx = 0;                 // 0 = the model's own context length
    std::string device = "auto";     // "auto", "cpu", "gpu", or a ggml device name ("Vulkan0", "CUDA0")
    int flash_attn = -1;             // 1 on, 0 off, -1 default (on, unless TIERLM_NO_FA is set)
    int kv_f32 = -1;                 // 1 f32 cache, 0 f16, -1 default (f16, unless TIERLM_KV_F32 is set)
    int batch = 0;                   // tokens per graph when reading a prompt; 0 = 256
    int experts_used = 0;            // override active experts per token (MoE); 0 = the model's value
    int vram_budget_mb = 0;          // most VRAM to use for weights + KV cache; 0 = whatever is free
    // How to split a model that doesn't fit in VRAM:
    //   "layers"   whole layers: the last ones (with their KV cache) in VRAM, the first ones in RAM.
    //              One GPU<->CPU hand-off per token; usually fastest, especially on Windows.
    //   "experts"  attention, router and KV cache of every layer in VRAM, then expert layers while they
    //              fit. Two hand-offs per RAM layer per token; can win on long contexts.
    std::string split_mode = "layers";
};

// Where the time goes, accumulated over every eval_batch call (for profiling).
struct EvalStats {
    int calls = 0, tokens = 0;
    double build_ms = 0, alloc_ms = 0, input_ms = 0, compute_ms = 0, output_ms = 0;
    int splits = 0, copies = 0;                  // of the last graph: GPU<->CPU hand-offs and tensor copies
};

// Where the tier loader put things.
struct Placement {
    uint64_t gpu_weights = 0, cpu_weights = 0;   // bytes of weights in VRAM and in system RAM
    bool kv_on_gpu = false;                      // the whole KV cache is in VRAM
    int kv_gpu_layers = 0;                       // layers whose KV cache is in VRAM
    std::string mode;                            // split mode used
    int layers = 0;
    int attn_gpu = 0;                            // layers whose attention weights are in VRAM
    int ffn_gpu = 0;                             // layers whose feed-forward / experts are in VRAM
    bool split() const { return gpu_weights > 0 && cpu_weights > 0; }
};

// Forward pass through ggml's backend API, so it runs on any ggml device (CPU, Vulkan, CUDA, ...).
// All weights and the KV cache live on one device for now; the tier loader will later spread them out.
class Model {
public:
    // device: "auto" (the biggest dedicated GPU, else an integrated one, else the CPU), "cpu", "gpu",
    // or a specific ggml device name such as "Vulkan0" or "CUDA0".
    static std::unique_ptr<Model> load(const std::string& path, const LoadOptions& opt);   // throws on failure
    static std::unique_ptr<Model> load(const std::string& path, int max_ctx = 0, const std::string& device = "auto") {
        LoadOptions o; o.max_ctx = max_ctx; o.device = device; return load(path, o);
    }
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Appends `ids` after the tokens already in the cache; returns logits for the last one.
    // n_threads applies to whatever runs on the CPU. If `routing` is given (MoE models), it receives the
    // experts the router picked: n_tok x n_layer x n_expert_used ids, best first.
    std::vector<float> eval(const std::vector<int32_t>& ids, int n_threads, std::vector<int32_t>* routing = nullptr);
    void reset() { n_past_ = 0; }                 // forget the conversation
    void truncate(int n) { if (n < n_past_) n_past_ = n < 0 ? 0 : n; }   // keep only the first n tokens
    int n_past() const { return n_past_; }
    int max_ctx() const { return max_ctx_; }
    const std::string& device_name() const { return device_name_; }
    const std::string& arch() const { return hp_.arch; }
    uint64_t weights_bytes() const { return weights_bytes_; }
    const Placement& placement() const { return place_; }
    const EvalStats& stats() const { return stats_; }
    void reset_stats() { stats_ = EvalStats(); }
    uint64_t kv_bytes() const { return kv_bytes_; }
    bool on_gpu() const { return main_ != cpu_; }
    bool integrated_gpu() const;                  // shares system memory (e.g. Intel Arc in a laptop)
    void device_memory(size_t& free, size_t& total) const;   // of the device holding the weights
    bool flash_attn() const { return flash_attn_; }
    const char* kv_type() const { return kv_f32_ ? "f32" : "f16"; }
    int batch() const { return batch_; }
    int experts_default() const { return n_expert_used_file_; }   // active experts the model was trained with

    // Convenience: reset, then evaluate the whole sequence.
    std::vector<float> forward(const std::vector<int32_t>& ids, int n_threads) { reset(); return eval(ids, n_threads); }

    const Hparams& hparams() const { return hp_; }

private:
    Model() = default;
    struct Graph {
        ggml_context* ctx = nullptr;
        struct ggml_cgraph* gf = nullptr;
        ggml_tensor *inp = nullptr, *pos = nullptr, *mask = nullptr, *logits = nullptr;
        std::vector<ggml_tensor*> orders;        // router rankings per layer (when routing is wanted)
        std::vector<std::pair<ggml_tensor*, int>> owner;   // every op tensor of each layer, with its layer
    };
    Graph build(int n_tok, int n_past, bool want_routing);
    void pin_cpu_layers(const Graph& g);        // keep every op of a RAM layer on the CPU
    std::vector<float> eval_batch(const int32_t* ids, int n_tok, int n_threads, std::vector<int32_t>* routing);
    struct Layer {
        ggml_tensor *attn_norm = nullptr, *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
        ggml_tensor *bq = nullptr, *bk = nullptr, *bv = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        ggml_tensor *ffn_norm = nullptr;
        ggml_tensor *gate = nullptr, *up = nullptr, *down = nullptr;          // dense FFN
        ggml_tensor *router = nullptr, *gate_x = nullptr, *up_x = nullptr, *down_x = nullptr;   // experts
        ggml_tensor *k_cache = nullptr, *v_cache = nullptr;   // [head_dim, n_head_kv, max_ctx]
    };
    Hparams hp_;
    ggml_backend* main_ = nullptr;               // device holding weights + cache
    ggml_backend* cpu_ = nullptr;                // fallback for ops the device can't run
    ggml_backend_sched* sched_ = nullptr;
    ggml_context* wctx_ = nullptr;               // weight tensor descriptors
    ggml_context* kvctx_ = nullptr;              // KV cache tensor descriptors
    std::vector<ggml_backend_buffer*> wbufs_;    // weight memory: one buffer in VRAM, one in system RAM
    Placement place_;
    EvalStats stats_;
    std::vector<ggml_backend_buffer*> kvbufs_;   // KV cache memory: VRAM part and RAM part
    std::vector<uint8_t> graph_mem_;             // scratch for building graphs
    ggml_tensor *tok_embd_ = nullptr, *out_norm_ = nullptr, *output_ = nullptr, *rope_freqs_ = nullptr;
    std::vector<Layer> layers_;
    std::vector<bool> layer_kv_gpu_;             // per layer: KV cache in VRAM
    bool pin_layers_ = false;                    // layer split: RAM layers must run entirely on the CPU
    std::string device_name_;
    int max_ctx_ = 0, n_past_ = 0;
    uint64_t weights_bytes_ = 0, kv_bytes_ = 0;
    void* dev_ = nullptr;                        // ggml_backend_dev_t of the main device
    bool flash_attn_ = true, kv_f32_ = false;
    int batch_ = 256, n_expert_used_file_ = 0;
};
using OlmoeModel = Model;

}
