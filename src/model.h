#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <array>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_buffer_type;
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
//   gpt-oss   OpenAI gpt-oss (20B, 120B, and derivatives): MoE with biases everywhere, attention sinks, every
//             other layer attending to the last 128 tokens only, YaRN positions, clamped SwiGLU experts.
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
    bool expert_softmax_first = true; // softmax over all experts, then pick; false: pick, then softmax over the picked
    bool expert_sigmoid = false;      // Flint: each expert scored by a sigmoid (no softmax), then the top ones picked
    int n_dense_lead = 0;             // Flint: the first layers have a dense feed-forward block instead of experts
    std::vector<bool> rope_layer;     // per layer: rotates positions (empty: all do). Flint: only sliding-window layers
    bool swiglu_oai = false;
    bool expert_bias = false;         // experts have biases (gpt-oss): no hot-expert split yet          // gpt-oss experts: clamped SwiGLU, out = (clamp(up) + 1) * g * sigmoid(1.702 g), g = min(gate, 7)
    int n_swa = 0;                    // sliding-window attention: these layers see only the last n_swa tokens
    std::vector<bool> swa;            // per layer: uses the window
    bool is_swa(int il) const { return n_swa > 0 && il < (int)swa.size() && swa[il]; }
    // YaRN context extension, computed as Hugging Face does: per-dimension frequency factors (ggml divides each
    // rotation frequency by its factor) and the scale of cos/sin. Empty: no YaRN.
    std::vector<float> rope_ff;
    // Qwen3.5: positions in 3 parts (image/time, row, column), interleaved over the rotated dimensions in these
    // sizes (pairs of dimensions). Text has all three equal, which is plain RoPE; images differ (see add_image).
    bool mrope = false;
    int mrope_sections[4] = {0, 0, 0, 0};
    float rope_attn_factor = 1.0f;
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
    int max_ctx = 0;                 // 0 = the model's own context length (shared by all conversations, see parallel)
    // Conversations the model can hold and compute at once (eval_seqs). They share the max_ctx cache positions,
    // each taking what it needs; DeltaNet layers keep one small running memory per conversation.
    int parallel = 1;
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
    // Hot experts: how often each expert has been picked, n_layer x n_expert (from earlier use of this model). With
    // experts in RAM, the most-used experts of each layer then go in VRAM ahead of whole expert layers, so most
    // picks are served from fast memory. Empty: whole expert layers, as before.
    std::vector<float> expert_usage;
    // Live hot experts: start from the busiest experts as above, then keep adjusting while the model runs: an
    // expert a conversation keeps picking moves into VRAM in place of one it stopped using (copied from
    // RAM between steps, a few per step). Needs expert_usage. The RAM part keeps a copy of every expert (more RAM:
    // the VRAM experts are in both), so any expert can come or go without reading the file again.
    bool live_hot = false;
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
    std::vector<std::vector<int>> hot;           // per layer: experts kept in VRAM by the hot-expert placement (all of
                                                 // them when ffn_vram is set; empty when none or not used)
    int hot_experts = 0;                         // experts in VRAM through the hot-expert placement, all layers
    // Every layer (attention, feed-forward/experts and its cache) in VRAM: only the token embedding table, which
    // is looked up by row and costs nothing per step, may be in RAM.
    bool all_on_gpu() const { return gpu_weights > 0 && layers > 0 && attn_gpu == layers && ffn_gpu == layers && kv_on_gpu; }
    // Work is shared between GPU and CPU (some layers or experts in RAM).
    bool split() const { return gpu_weights > 0 && cpu_weights > 0 && !all_on_gpu(); }
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

    // A conversation's memory, copied out to system RAM so another conversation can use the cache meanwhile and
    // this one can come back without being read again: the keys and values of its n_past positions in every
    // attention layer (and the MTP draft block), and the running memories of DeltaNet layers.
    struct Snapshot { int n_past = 0; std::vector<std::vector<uint8_t>> parts; size_t bytes = 0;
                      std::vector<std::array<int, 3>> rope; };   // (images) each position's 3-part position
    Snapshot save() const;
    // Live hot experts (LoadOptions::live_hot): how well VRAM is serving the router's picks.
    struct LiveStats { long long picks = 0, hits = 0, swaps = 0; double swap_ms = 0; };
    const LiveStats& live_stats() const { return live_stats_; }
    void reset_live_stats() { live_stats_ = LiveStats(); }
    bool live_hot() const { return !live_.empty(); }
    void restore(const Snapshot& s);          // the cache becomes exactly the saved conversation
    int n_past() const { return n_past_; }
    int max_ctx() const { return max_ctx_; }
    // Change the context length (all conversations' positions together) without loading the model again: the
    // cache is made again at the new size on the same devices, keeping what's in it. Throws if the conversations
    // in the cache are longer than the new size, or if it doesn't fit (the old size is then kept, unchanged).
    // Not while a step runs (the caller holds the engine's lock).
    void resize_ctx(int new_ctx);
    int ctx_in_use() const { return hi_; }        // one past the highest cache position any conversation uses
    // Bytes the KV cache takes per position, all layers (the hybrid models' fixed running memories aside), and how
    // much of that is in VRAM (the rest is in RAM).
    size_t kv_bytes_per_token(size_t* in_vram = nullptr) const;
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

    // ---- several conversations at once ----
    // Conversation 0 is the one eval / eval_ex / save / restore / the MTP block work on. With LoadOptions::parallel
    // > 1, conversations 1 .. n_seq_max()-1 exist as well, and eval_seqs advances any of them (0 included) in one
    // step: the weights are read once for all of them. Each conversation has its own positions; their keys and
    // values share the cache (each position goes to any free place in it, and the attention mask keeps every
    // conversation to its own). Returns the logits of each input's last token, in input order.
    struct SeqInput { int seq = 0; std::vector<int32_t> ids; };

    // ---- images (Qwen3.5 with its vision file) ----
    // An image's image words (Vision::encode: rows x cols vectors of n_embd numbers, row by row) are registered
    // once and then fed like tokens, as ids first .. first + rows*cols - 1 (all >= kImageIdBase), between the
    // model's vision start and end tokens. Each word gets the image's 3-part position (start, start + row,
    // start + column); text after it continues from start + max(rows, cols). key identifies the image (a hash
    // of its pixels), so sending the same image again gives the same ids and reuses cached conversations.
    // Images not used for a while are forgotten when they take more than about 1 GB.
    static constexpr int32_t kImageIdBase = 1 << 30;
    static bool is_image_id(int32_t id) { return id >= kImageIdBase; }
    int32_t add_image(uint64_t key, std::vector<float> words, int rows, int cols);
    int32_t find_image(uint64_t key);            // its first id, or -1
    bool has_positions3() const { return hp_.mrope; }
    std::vector<std::vector<float>> eval_seqs(const std::vector<SeqInput>& in, int n_threads, std::vector<int32_t>* routing = nullptr);
    int n_seq_max() const { return (int)seq_cells_.size(); }
    int seq_past(int s) const { return (int)seq_cells_[s].size(); }
    void seq_reset(int s);                        // forget conversation s (s = 0: like reset())
    void seq_truncate(int s, int n);              // keep its first n tokens (hybrid models: all or nothing)
    int free_cells() const { return n_free_; }    // cache positions not used by any conversation

    // Convenience: reset, then evaluate the whole sequence.
    std::vector<float> forward(const std::vector<int32_t>& ids, int n_threads) { reset(); return eval(ids, n_threads); }

    const Hparams& hparams() const { return hp_; }

private:
    Model() = default;
    struct Graph {
        ggml_context* ctx = nullptr;
        struct ggml_cgraph* gf = nullptr;
        ggml_tensor *inp = nullptr, *pos = nullptr, *mask = nullptr, *logits = nullptr;
        ggml_tensor *img = nullptr, *keep = nullptr;   // images: image words [n_embd, n_tok], 1 for text tokens [1, n_tok]
        bool mrope = false;                      // pos holds 4 x n_tok 3-part positions
        ggml_tensor *hidden = nullptr;           // final-norm hidden states of every position (for the MTP head)
        ggml_tensor *mask_swa = nullptr;         // causal mask limited to the sliding window (models that use one)
        ggml_tensor *rope_ff = nullptr;          // YaRN frequency factors (Hparams::rope_ff), as input
        ggml_tensor *h_in = nullptr;             // MTP head: the target model's hidden states, as input
        ggml_tensor *cells = nullptr;            // where each token's key and value go in the cache
        ggml_tensor *out_idx = nullptr;          // several conversations: the tokens whose logits are wanted
        int n_tok = 0;
        // Per layer (DeltaNet layers only): for dropping the last d tokens, rollback[layer][d] = {memory, conv inputs}
        std::map<int, std::vector<std::pair<ggml_tensor*, ggml_tensor*>>> rollback;
        std::vector<ggml_tensor*> orders;        // each layer's chosen experts (when routing is wanted)
        std::vector<int> order_layers;           // the layer each of orders belongs to
        ggml_tensor* routing = nullptr;          // all of them joined: [n_used, n_tok * n_layer], one read
        std::vector<std::pair<ggml_tensor*, int>> owner;   // every op tensor of each layer, with its layer
        std::vector<ggml_tensor*> cpu_ops;       // small ops that must run on the CPU (hot-expert bookkeeping)
    };
    // A batch's tokens, conversation by conversation (contiguous): n tokens of conversation seq.
    struct SeqGroup { int seq, n; };
    // n_kv_end: the cache positions attention covers (all used ones), at least n_tok.
    Graph build(int n_tok, int n_kv_end, bool want_routing, int n_logits = 1, bool want_hidden = false, int snapshots = 1,
                const std::vector<SeqGroup>* groups = nullptr);
    struct Layer;
    ggml_tensor* attn_block(ggml_context* c, Graph& g, const Layer& L, ggml_tensor* h, int n_tok, int n_past);
    ggml_tensor* ffn_block(ggml_context* c, Graph& g, const Layer& L, ggml_tensor* h, int n_tok, bool want_routing);
    void pin_cpu_layers(const Graph& g);        // keep every op of a RAM layer on the CPU
    std::vector<float> eval_batch(const int32_t* ids, int n_tok, int n_threads, std::vector<int32_t>* routing);
    void release_pending();
    // One graph: n_tok tokens of conversation 0 (groups null), or of the conversations in groups.
    EvalOut step(const int32_t* ids, int n_tok, int n_threads, int n_logits, bool want_hidden, int snapshots,
                 std::vector<int32_t>* routing, const std::vector<SeqGroup>* groups);
    Graph build_mtp(int n_tok, int n_past);
    void* pending_ = nullptr;                    // Graph of the last undoable batch, until accept()
    // Live hot experts: per split layer, which expert each VRAM slot holds and how much each is wanted.
    struct LivePart { ggml_tensor *hot, *cold; };   // an expert tensor's VRAM part and its RAM part (all experts)
    struct LiveLayer {
        int layer = 0, H = 0;
        std::vector<int> slot_of, in_slot;       // expert -> VRAM slot (-1: not there); slot -> expert
        std::vector<float> score;                // learned use, then recent picks (fading)
        std::vector<LivePart> parts;
        ggml_tensor *map_hot = nullptr, *map_cold = nullptr, *mask = nullptr;
    };
    std::vector<LiveLayer> live_;
    mutable LiveStats live_stats_;
    void adapt_hot(const std::vector<int32_t>& routing, int n_tok);
    struct Layer {
        ggml_tensor *attn_norm = nullptr, *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
        ggml_tensor *bq = nullptr, *bk = nullptr, *bv = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        ggml_tensor *bo = nullptr, *sinks = nullptr;                          // gpt-oss: output bias, attention sinks
        ggml_tensor *router_b = nullptr, *gate_xb = nullptr, *up_xb = nullptr, *down_xb = nullptr;   // gpt-oss: biases
        bool rope = true;                        // rotates positions (see Hparams::rope_layer)
        int index = 0;                           // the layer's number
        bool swa = false;                                                     // sliding-window layer
        ggml_tensor *ffn_norm = nullptr;
        ggml_tensor *gate = nullptr, *up = nullptr, *down = nullptr;          // dense FFN
        ggml_tensor *router = nullptr, *gate_x = nullptr, *up_x = nullptr, *down_x = nullptr;   // experts
        ggml_tensor *gate_up_x = nullptr;                                     // gate+up experts in one tensor
        // Hot-expert split: the layer's busiest experts in VRAM (..._h, plus one all-zero expert at the end), the rest
        // in RAM (..._c, likewise). The router's picks are mapped to each part's own numbering; a pick that lives
        // in the other part goes to the zero expert with weight 0.
        ggml_tensor *gate_xh = nullptr, *up_xh = nullptr, *down_xh = nullptr, *gate_up_xh = nullptr;
        ggml_tensor *gate_xc = nullptr, *up_xc = nullptr, *down_xc = nullptr, *gate_up_xc = nullptr;
        ggml_tensor *map_hot = nullptr, *map_cold = nullptr, *hot_mask = nullptr;   // [1, n_expert]
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
        std::vector<ggml_tensor*> conv_states, ssm_states;   // per conversation ([0]: conv_state, ssm_state_t)
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
    std::vector<ggml_backend_buffer_type*> kv_bufts_;   // where each part of the cache goes: per GPU, then RAM
    void make_kv(int ctx, int n_seq);
    void free_kv();
    // Scratch for building graphs, one per kind (one token; several tokens; the MTP draft block): the GPU keeps a
    // recorded replay (CUDA graph) per graph, found by where it was built, so each kind keeps its own.
    std::vector<uint8_t> graph_mem_[3];
    std::vector<uint8_t>& graph_mem(int kind) { return graph_mem_[kind]; }
    ggml_tensor *tok_embd_ = nullptr, *out_norm_ = nullptr, *output_ = nullptr, *rope_freqs_ = nullptr;
    std::vector<Layer> layers_;
    std::unique_ptr<Layer> mtp_;                 // the MTP draft block, when loaded
    std::vector<bool> layer_kv_gpu_;             // per layer: KV cache in VRAM
    std::vector<int> layer_dev_;                 // per layer: the GPU (index into gpus_) holding its KV cache, or -1 for RAM
    struct { int graphs = 0; double tokens = 0, prep = 0, compute = 0, out = 0; } prof_;   // TIERLM_PROFILE
    // Attention covers the cache in steps of 256 positions (the extra ones are masked out), so the graph keeps the
    // same shapes from one token to the next and the GPU can replay it (CUDA graphs) instead of launching each op.
    int kv_span(int n) const { const int s = (n + 255) / 256 * 256; return s < max_ctx_ ? s : (n > max_ctx_ ? n : max_ctx_); }
    void profile_step(int n_tok, double prep_ms, double compute_ms, double out_ms);
    bool pin_layers_ = false;                    // layer split: RAM layers must run entirely on the CPU
    std::string device_name_;
    int max_ctx_ = 0, n_past_ = 0;               // n_past_: conversation 0's tokens (seq_cells_[0].size())
    // The cache's positions ("cells"): which conversation uses each and for which of its positions (-1: free);
    // per conversation, the cell of each of its positions. Cells are handed out lowest first, so a single
    // conversation's positions are its cells, as before.
    std::vector<int> cell_seq_, cell_pos_;
    std::vector<std::vector<int>> seq_cells_;
    // (Qwen3.5) per conversation: each position's 3-part position, and the next text position
    std::vector<std::vector<std::array<int, 3>>> seq_rope_;
    std::vector<int> seq_next_;
    void rope_truncate(int s, int n);
    struct ImageRec { uint64_t key = 0; int32_t first = 0; int rows = 0, cols = 0; std::vector<float> words; uint64_t used = 0; };
    std::vector<ImageRec> images_;
    int32_t next_image_id_ = kImageIdBase;
    uint64_t image_clock_ = 0;
    const ImageRec* image_of(int32_t id) const;
    bool build_images_ = false, build_mrope_ = false;   // set by step() for build()
    int n_free_ = 0, graph_size_ = 16384, hi_ = 0, free_lo_ = 0;   // hi_: one past the highest used cell
    std::vector<int> alloc_cells(int s, int n);   // n more positions for conversation s; throws when full
    void free_cells_from(int s, int n);          // conversation s keeps its first n positions
    int cells_end() const;                       // one past the highest cell in use
    void fill_masks(Graph& g, int n_kv, const std::vector<int>& tok_seq, const std::vector<int>& tok_pos);
    void clear_state(int s);                     // DeltaNet memories of conversation s back to zero
    uint64_t weights_bytes_ = 0, kv_bytes_ = 0;
    void* dev_ = nullptr;                        // ggml_backend_dev_t of the main device
    bool flash_attn_ = true, kv_f32_ = false;
    int kv_t_ = 1;                               // ggml_type of the KV cache (GGML_TYPE_F16)
    int batch_ = 256, n_expert_used_file_ = 0;
};
using OlmoeModel = Model;

}
