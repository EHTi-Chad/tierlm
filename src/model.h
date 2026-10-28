#pragma once
#include <cstdint>
#include <functional>
#include <map>
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
//   qwen3next Qwen3-Next / Qwen3-Coder-Next (80B-A3B): the hybrid's predecessor, with differently stored weights
//   qwen35moe Qwen3.5 / 3.6 MoE (e.g. Qwen3.6-35B-A3B): the same hybrid, with experts plus a shared expert
//   qwen35    Qwen3.5 / 3.6 / 3.8 dense: a hybrid. Three of every four layers use Gated DeltaNet linear
//             attention (a fixed-size running memory instead of a KV cache); the rest use gated full
//             attention with partial RoPE.
bool arch_supported(const std::string& arch);
// A separate MTP draft file ("mtp-*.gguf") in the same folder that matches this model, or "".
std::string mtp_companion(const std::string& model_path);
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
    float expert_scale = 0.0f;        // multiply the chosen experts' weights by this (0 or 1 = leave as is)
    bool expert_softmax_first = true; // softmax over all experts, then pick (vs pick then softmax: same order)
    bool is_moe() const { return n_expert > 0; }
    // Hybrid (qwen35): Gated DeltaNet layers
    bool hybrid = false;
    std::vector<bool> recurrent;      // per layer: linear attention (true) or full attention (false)
    int ssm_conv = 0;                 // causal convolution width (4)
    int ssm_state = 0;                // head size of the linear attention (128)
    int ssm_k_heads = 0, ssm_v_heads = 0;
    bool v_grouped = false;
    int n_nextn = 0;                  // multi-token-prediction draft blocks in the file (after the n_layer layers)           // value heads grouped by key head (Qwen3-Next) instead of tiled (Qwen3.5)
    bool is_recurrent(int il) const { return hybrid && il < (int)recurrent.size() && recurrent[il]; }
    int ssm_conv_dim() const { return 2 * ssm_k_heads * ssm_state + ssm_v_heads * ssm_state; }
};
using OlmoeHparams = Hparams;

// Load-time options. -1 / 0 mean "use the default".
struct LoadOptions {
    int max_ctx = 0;                 // 0 = the model's own context length
    std::string device = "auto";     // "auto", "cpu", "gpu", or a ggml device name ("Vulkan0", "CUDA0")
    int flash_attn = -1;             // 1 on, 0 off, -1 default (on, unless TIERLM_NO_FA is set)
    int kv_f32 = -1;                 // 1 f32 cache, 0 f16, -1 default (f16, unless TIERLM_KV_F32 is set)
    std::string kv_type;             // "f16", "f32", "q8_0" or "q4_0"; overrides kv_f32 when set
    int batch = 0;                   // tokens per graph when reading a prompt; 0 = 256
    int experts_used = 0;            // override active experts per token (MoE); 0 = the model's value
    int vram_budget_mb = 0;          // most VRAM to use for weights + KV cache; 0 = whatever is free
    // How to split a model that doesn't fit in VRAM:
    //   "layers"   whole layers: the last ones (with their KV cache) in VRAM, the first ones in RAM.
    //              One GPU<->CPU hand-off per token; usually fastest, especially on Windows.
    //   "experts"  attention, router and KV cache of every layer in VRAM, then expert layers while they
    //              fit. Two hand-offs per RAM layer per token. Much faster for MoE models: on a laptop RTX 3070,
    //              Qwen3.6-35B-A3B went from 9.9 to 17 tok/s.
    //   "auto"     "experts" for mixture-of-experts models, "layers" for the rest (the default).
    std::string split_mode = "auto";
    bool repack = true;              // store CPU-side weights in ggml's faster repacked layout where it has one
    bool mtp = false;                // also load the MTP draft block, for speculative decoding (if the file has one)
    // With device "auto" or "gpu": also use the other GPUs of the same kind (several NVIDIA cards with CUDA, say)
    // as further VRAM tiers, filled after the first one and before system RAM.
    bool multi_gpu = true;
    std::function<void(float)> progress;   // called while the weights are read, with the share done (0..1)
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
    int repacked = 0;                            // CPU-side weight tensors stored in the repacked layout
    struct Gpu { std::string name; uint64_t weights = 0, kv = 0; int layers = 0; };
    std::vector<Gpu> gpus;                       // per GPU (more than one with several cards): what it holds
    std::vector<bool> ffn_vram;                  // per layer: its feed-forward part (the experts, in MoE models) is in VRAM
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
    void reset();                                 // forget the conversation

    // ---- speculative decoding ----
    struct EvalOut {
        std::vector<float> logits;                // n_logits x n_vocab: predictions for the last n_logits positions
        std::vector<float> hidden;                // n_tok x n_embd final-norm hidden states (if asked for)
    };
    // Like eval, for one batch of up to batch() tokens, with more outputs. snapshots > 1 makes the batch undoable:
    // accept(n_keep) then keeps only its first n_keep tokens (snapshots must be at least the batch size).
    EvalOut eval_ex(const int32_t* ids, int n_tok, int n_threads, int n_logits, bool want_hidden, int snapshots,
                    std::vector<int32_t>* routing = nullptr);
    void accept(int n_keep);
    // The MTP draft block (when loaded with LoadOptions::mtp): ids[i] at position pos0 + i, paired with the
    // final hidden state h[i] of the position before it. Writes the block's KV cache at those positions.
    // Returns its prediction for the position after the last one; h_out gets its hidden states (n_tok x n_embd).
    bool has_mtp() const { return mtp_ != nullptr; }
    std::vector<float> mtp_eval(const int32_t* ids, const float* h, int n_tok, int pos0, int n_threads,
                                std::vector<float>* h_out = nullptr);
    // Keep only the first n tokens. Hybrid models can't rewind their running memory, so for them anything
    // short of the whole conversation means starting over (n_past() tells the caller where it stands).
    void truncate(int n);
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
    const char* kv_type() const;      // "f16", "f32", "q8_0" or "q4_0"
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
        ggml_tensor *hidden = nullptr;           // final-norm hidden states of every position (for the MTP head)
        ggml_tensor *h_in = nullptr;             // MTP head: the target model's hidden states, as input
        int n_tok = 0;
        // Per layer (DeltaNet layers only): for dropping the last d tokens, rollback[layer][d] = {memory, conv inputs}
        std::map<int, std::vector<std::pair<ggml_tensor*, ggml_tensor*>>> rollback;
        std::vector<ggml_tensor*> orders;        // router rankings per layer (when routing is wanted)
        std::vector<std::pair<ggml_tensor*, int>> owner;   // every op tensor of each layer, with its layer
    };
    Graph build(int n_tok, int n_past, bool want_routing, int n_logits = 1, bool want_hidden = false, int snapshots = 1);
    struct Layer;
    ggml_tensor* attn_block(ggml_context* c, Graph& g, const Layer& L, ggml_tensor* h, int n_tok, int n_past);
    ggml_tensor* ffn_block(ggml_context* c, Graph& g, const Layer& L, ggml_tensor* h, int n_tok, bool want_routing);
    void pin_cpu_layers(const Graph& g);        // keep every op of a RAM layer on the CPU
    std::vector<float> eval_batch(const int32_t* ids, int n_tok, int n_threads, std::vector<int32_t>* routing);
    void release_pending();
    Graph build_mtp(int n_tok, int n_past);
    void* pending_ = nullptr;                    // Graph of the last undoable batch, until accept()
    struct Layer {
        ggml_tensor *attn_norm = nullptr, *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
        ggml_tensor *bq = nullptr, *bk = nullptr, *bv = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        ggml_tensor *ffn_norm = nullptr;
        ggml_tensor *gate = nullptr, *up = nullptr, *down = nullptr;          // dense FFN
        ggml_tensor *router = nullptr, *gate_x = nullptr, *up_x = nullptr, *down_x = nullptr;   // experts
        ggml_tensor *gate_up_x = nullptr;                                     // gate+up experts in one tensor
        ggml_tensor *up_s = nullptr, *gate_s = nullptr, *down_s = nullptr, *gate_inp_s = nullptr;   // shared expert
        ggml_tensor *k_cache = nullptr, *v_cache = nullptr;   // [head_dim, n_head_kv, max_ctx]
        // Hybrid models: Gated DeltaNet layer
        bool recurrent = false;
        ggml_tensor *wqkv = nullptr, *wgate = nullptr, *ssm_conv1d = nullptr, *ssm_dt = nullptr, *ssm_a = nullptr;
        ggml_tensor *ssm_beta = nullptr, *ssm_alpha = nullptr, *ssm_norm = nullptr, *ssm_out = nullptr;
        ggml_tensor *ssm_ba = nullptr, *ssm_in = nullptr;    // Qwen3-Next: fused beta/alpha, and older fused q/k/v/z
        ggml_tensor *conv_state = nullptr;       // [ssm_conv - 1, conv_dim]: the last few inputs
        // MTP draft block only
        ggml_tensor *eh_proj = nullptr, *enorm = nullptr, *hnorm = nullptr, *mtp_embd = nullptr, *head_norm = nullptr, *head = nullptr;
        ggml_tensor *ssm_state_t = nullptr;      // [S, S, H_v]: the running memory
    };
    Hparams hp_;
    ggml_backend* main_ = nullptr;               // device holding weights + cache (the first GPU)
    std::vector<ggml_backend*> gpus_;            // every GPU in use, main_ first
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
    std::unique_ptr<Layer> mtp_;                 // the MTP draft block, when loaded
    std::vector<bool> layer_kv_gpu_;             // per layer: KV cache in VRAM
    std::vector<int> layer_dev_;                 // per layer: the GPU (index into gpus_) holding its KV cache, or -1 for RAM
    bool pin_layers_ = false;                    // layer split: RAM layers must run entirely on the CPU
    std::string device_name_;
    int max_ctx_ = 0, n_past_ = 0;
    uint64_t weights_bytes_ = 0, kv_bytes_ = 0;
    void* dev_ = nullptr;                        // ggml_backend_dev_t of the main device
    bool flash_attn_ = true, kv_f32_ = false;
    int kv_t_ = 1;                               // ggml_type of the KV cache (GGML_TYPE_F16)
    int batch_ = 256, n_expert_used_file_ = 0;
};
using OlmoeModel = Model;

}
