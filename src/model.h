#pragma once
#include <chrono>
#include "mapped_file.h"
#include "ngram.h"
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
struct ggml_backend_event;

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
    // Gemma 3 / 4: the embedding scaled by sqrt(n_embd); a norm after attention and after the feed-forward block too
    // (before each is added back); GELU (tanh) in the feed-forward block; sliding-window layers rotate positions with
    // their own base (rope_base_swa), the full ones with rope_base and the file's scaling
    bool gemma = false;
    float embd_scale = 0.0f;          // 0: none
    float rope_base_swa = 0.0f;       // 0: as the other layers
    bool gelu_ffn = false;
    float final_softcap = 0.0f;       // logits = cap tanh(logits / cap) (0: none)
    float attn_softcap = 0.0f;        // the same on the attention scores
    // Gemma 4: per layer, the head size, the key/value heads and the rotated size (sliding and full layers differ;
    // empty: head_dim, n_head_kv, n_rot everywhere). Full layers without their own value projection use the key's
    // (before its norm); values are normed (no weight); attention scale 1. The layers from kv_from on keep no cache
    // of their own: each reads that of the last layer before kv_from of its kind (sliding or full). The 26B A4B's
    // layers run a dense MLP and the experts side by side (see build). E2B / E4B: per-layer inputs (g4_ple wide).
    bool gemma4 = false;
    std::vector<int> hd_l, kvh_l, rot_l;
    int kv_from = 0;
    int g4_ple = 0;
    std::vector<int> nh_l;            // per layer: the query heads (empty: n_head everywhere; Inkling 2's sliding layers)
    int nh_at(int il) const { return il >= 0 && il < (int)nh_l.size() && nh_l[il] ? nh_l[il] : n_head; }
    // Inkling 2 (Thinking Machines; "inkling2", "inkling2_loop"): no RoPE. Each attention layer's queries are scaled
    // per head and clipped to a length (qk_clip), keys clipped too, values normed; a bias by distance is read from each
    // token's own relative states (rel_dim per head, scaled and clipped to rel_clip, projected on a table of
    // rel_extent distances: the window's size in the sliding layers); one more score per token and head (a sink, zero
    // value); the output gated by sigmoid of the input. The full-attention layers scale queries and relative states by
    // 1 + log_alpha log(max(1, (pos + 1) / log_floor)). Short causal convolutions (conv taps) on the keys and values
    // (full layers: plus taps that depend on the token, per key/value head) and on both blocks' outputs, each with its
    // input added back and scaled. Instead of a plain residual, each layer reads the earlier blocks' outputs (a block
    // is res_block layers) mixed by attention with learned queries. Experts: sigmoid scores (a bias for choosing),
    // weights renormalised and scaled; softcapped SwiGLU, gate capped at situ_beta, up at situ_lbeta. Logits capped,
    // the padded vocabulary (output rows past n_vocab_real) dropped. The loop: layers [loop_lo, loop_hi) run
    // loop_passes times, each later pass from the same input, with messages from the previous pass (its last blocks
    // and output, mixed by 4 learned queries per layer) mixed into the queries', keys', values' and experts' inputs
    // through low-rank gates; each pass keeps its own cache (see Model::ink_slots_).
    bool ink = false;
    int ink_conv = 0, ink_rel_dim = 0, ink_rel_extent = 0, ink_res_block = 0, ink_log_floor = 0;
    float ink_qk_clip = 0, ink_rel_clip = 0, ink_log_alpha = 0, situ_beta = 0, situ_lbeta = 0;
    int ink_loop_lo = 0, ink_loop_hi = 0, ink_loop_passes = 1, ink_loop_rank = 0;
    bool ink_earlier = false;                 // the earlier layout of a looped file (arch "inkling2" with num_loops)
    int n_vocab_real = 0;             // logits kept (0: all the output rows); Inkling 2 pads its tables past it
    int ink_loop_n() const { return ink_loop_hi - ink_loop_lo; }
    // The convolutions' saved inputs per layer: keys, values (head width x key/value heads each), both outputs
    int ink_conv_ch(int il) const { return 2 * hd_at(il) * kvh_at(il) + 2 * n_embd; }
    int hd_at(int il) const { return il >= 0 && il < (int)hd_l.size() && hd_l[il] ? hd_l[il] : head_dim; }
    int kvh_at(int il) const { return il >= 0 && il < (int)kvh_l.size() && kvh_l[il] ? kvh_l[il] : n_head_kv; }
    int rot_at(int il) const { return il >= 0 && il < (int)rot_l.size() && rot_l[il] ? rot_l[il] : n_rot; }
    int kv_src(int il) const {
        if (kv_from <= 0 || il < kv_from || il >= n_layer) return -1;
        for (int j = kv_from - 1; j >= 0; --j) if (is_swa(j) == is_swa(il)) return j;
        return -1;
    }
    enum QkNorm { kNone, kFull, kPerHead } qk_norm = kNone;
    bool qkv_bias = false;            // biases on Q/K/V projections
    bool expert_renorm = false;       // renormalise the chosen experts' weights to sum to 1
    float expert_scale = 0.0f;        // multiply the chosen experts' weights by this (0 or 1 = leave as is)
    bool expert_softmax_first = true; // softmax over all experts, then pick; false: pick, then softmax over the picked
    bool expert_sigmoid = false;      // Flint: each expert scored by a sigmoid (no softmax), then the top ones picked
    int n_dense_lead = 0;             // Flint: the first layers have a dense feed-forward block instead of experts
    std::vector<bool> rope_layer;     // per layer: rotates positions (empty: all do). Flint: only sliding-window layers
    bool swiglu_oai = false;
    bool expert_bias = false;         // experts have biases (gpt-oss)          // gpt-oss experts: clamped SwiGLU, out = (clamp(up) + 1) * g * sigmoid(1.702 g), g = min(gate, 7)
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
    // DeepSeek V2/V3, Kimi K2, Moonlight (deepseek2): multi-head latent attention. Each token's keys and values are
    // kept as one small compressed vector (kv_lora numbers, normalised) plus a shared rotated key part (n_rot); the
    // cache holds just those, kv_lora + n_rot numbers per token per layer, and attention runs on them directly (each
    // head's query is turned into that space first). q_lora: queries are compressed too (0: one plain projection).
    bool mla = false;
    int q_lora = 0, kv_lora = 0, nope = 0, v_dim = 0;   // nope: the unrotated part of each head's key; v_dim: value size
    float kq_scale = 0.0f;            // attention scale (0: 1/sqrt(head_dim)); MLA: 1/sqrt(nope + n_rot), YaRN-scaled
    // Expert routing by groups (DeepSeek V3): experts come in n_group groups; only the topk_group best groups (by
    // the sum of their two best scores, or by their best score with softmax routing as in V2) may be picked from.
    int n_group = 0, topk_group = 0;
    bool group_sum2 = true;
    // Qwen3.8 Flash-Next (qwen4exp): hyper-connections. The residual stream is hc copies of the width (4 x 2560),
    // mixed into one input for each block and added back to every copy with its own weight. hc = 0: one stream.
    int hc = 0;
    // The width of the hidden state the MTP draft block reads: the final-norm state (n_embd), or with
    // hyper-connections the hc copies before the last stream mixer (hc x n_embd).
    int hidden_width() const { return hc ? hc * n_embd : n_embd; }
    // ...and its per-layer embedding (PLE) on some layers: each token's hashed 2- and 3-grams look up rows of one
    // big table (ple_dim numbers per row, ngram.heads() rows per token), gated by the streams and smoothed by a
    // dilated causal convolution (ple_kernel taps, ngram_size apart).
    std::vector<bool> ple_layer;
    int ple_dim = 0, ple_kernel = 4;
    NgramHash ngram;
    bool any_ple() const { for (bool b : ple_layer) if (b) return true; return false; }
    int ple_state_len() const { return (ple_kernel - 1) * ngram.ngram; }   // inputs the convolution keeps
    // QSA (Qwen Sparse Attention, Flash-Next's attention layers): past qsa_budget tokens, each query attends only to
    // the blocks of qsa_ratio tokens an indexer ranks highest. Within the budget it is plain attention.
    int qsa_budget = 0;
    std::vector<int> qsa_ratio;       // per layer: tokens per indexed block (0: not a QSA layer)
    int qsa_heads = 0, qsa_dim = 0;   // the indexer's query heads and their size (one key head)
    int32_t ple_image_tok = -1;       // the token an image's words count as in the n-grams
    bool is_moe() const { return n_expert > 0; }
    // Hybrid (qwen35): Gated DeltaNet layers
    bool hybrid = false;
    std::vector<bool> recurrent;      // per layer: linear attention (true) or full attention (false)
    int ssm_conv = 0;                 // causal convolution width (4)
    int ssm_state = 0;                // head size of the linear attention (128)
    int ssm_k_heads = 0, ssm_v_heads = 0;
    bool v_grouped = false;
    bool dn_sigmoid_gate = false;   // Flash-Next: DeltaNet's output gate is sigmoid(z), not Qwen3.5's silu(z)
    int n_nextn = 0;                  // multi-token-prediction draft blocks in the file (after the n_layer layers)           // value heads grouped by key head (Qwen3-Next) instead of tiled (Qwen3.5)
    bool is_recurrent(int il) const { return hybrid && il < (int)recurrent.size() && recurrent[il]; }
    bool gated_attn = false;          // Qwen3.5: attention layers project [q | output gate] per head
    // Nemotron 3.5 (nemotron_h_moe): every block is one thing behind one norm: a Mamba-2 mixer (the running memory),
    // plain attention with no position encoding, or LatentMoE experts. block[il] says which.
    enum Block { kBoth = 0, kMamba, kAttn, kFfn };
    bool mamba = false;
    std::vector<int> block;           // per layer (empty: every layer is attention or DeltaNet, then feed-forward)
    int block_kind(int il) const { return il >= 0 && il < (int)block.size() ? block[il] : kBoth; }
    // Mamba-2: inner width, heads (m_inner / m_heads wide each), state size per inner channel, B/C groups
    int m_inner = 0, m_heads = 0, m_state = 0, m_groups = 0;
    // LatentMoE: the experts work at this width (0: the model's), between a down and an up projection; blocks have
    // their own number of experts, expert width and experts per token (n_expert / n_expert_used: the largest)
    int latent = 0;
    std::vector<int> n_expert_l, n_expert_used_l;
    bool expert_relu2 = false;        // experts are up, relu squared, down (no gate)
    int n_expert_at(int il) const { return il >= 0 && il < (int)n_expert_l.size() && n_expert_l[il] ? n_expert_l[il] : n_expert; }
    int n_used_at(int il) const { return il >= 0 && il < (int)n_expert_used_l.size() && n_expert_used_l[il] ? n_expert_used_l[il] : n_expert_used; }
    // The running memory of a recurrent layer, per conversation: the convolution's last inputs and the state
    int ssm_conv_dim() const { return mamba ? m_inner + 2 * m_groups * m_state : 2 * ssm_k_heads * ssm_state + ssm_v_heads * ssm_state; }
    int64_t ssm_mem_elems() const { return mamba ? (int64_t)m_state * m_inner : (int64_t)ssm_state * ssm_state * ssm_v_heads; }
    // DeepSeek V4 (deepseek4): hyper-connections (d4_hc streams, mixed by weights made from the streams, the mixing
    // matrix made doubly stochastic by d4_hc_iters Sinkhorn steps); attention with one key/value head that is both
    // (head_dim wide), a learned sink per head, partial interleaved RoPE on the last n_rot dims (undone on the output),
    // a grouped low-rank output projection (d4_og groups of d4_ol); each token sees the last d4_win tokens as they are
    // plus compressed entries of the past: per layer d4_ratio 0 (none), 4 (CSA: overlapping windows, the indexer
    // picks d4_itopk of them per token) or 128 (HCA: all of them). Compressed layers rotate with base d4_cbase and
    // YaRN (rope_scale, d4_yarn_*); the others plain with rope_base. Experts: sqrt(softplus) scores, the first
    // d4_hash layers routed by a token -> experts table, SwiGLU with clamped inputs (per layer).
    bool dsv4 = false;
    std::vector<int> d4_ratio;
    int d4_hc = 0, d4_hc_iters = 20, d4_og = 0, d4_ol = 0, d4_win = 0, d4_ih = 0, d4_id = 0, d4_itopk = 0, d4_hash = 0;
    float d4_hc_eps = 1e-6f, d4_cbase = 160000.0f, d4_yarn_fast = 32.0f, d4_yarn_slow = 1.0f;
    int d4_yarn_orig = 0;
    std::vector<float> d4_clamp, d4_clamp_s;
    std::vector<float> d4_rope_ff;    // the compressed layers' YaRN, as per-dimension frequency factors (see rope_ff)
    int d4_ratio_at(int il) const { return il >= 0 && il < (int)d4_ratio.size() ? d4_ratio[il] : 0; }
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
    std::string kv_type;             // "f16", "f32", "q8_0", "q5_0" or "q4_0"; overrides kv_f32 when set
    int batch = 0;                   // tokens per graph when reading a prompt; 0 = 256
    int experts_used = 0;            // override active experts per token (MoE); 0 = the model's value
    int vram_budget_mb = 0;          // most VRAM to use for weights + KV cache; 0 = whatever is free
    // VRAM kept free for the GPU's working memory (compute buffers, scratch, the driver); 0 = what vram_note says, else
    // 768 MB. After loading, tierlm measures what it really needs; if that's more, it loads again with that (once):
    // VRAM that runs out mid-reply isn't an error on Windows, the driver moves memory to system RAM and the GPU slows
    // down many times over.
    int vram_reserve_mb = 0;
    std::string vram_note;           // a file where that measured need is kept for the next load ("" = not kept)
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
    // The SSD tier: when the weights left for RAM don't fit in it, the least-used expert layers (then, for models
    // without experts, the first layers) are read from a drive as the model needs them instead of loaded (a memory
    // map, see MappedFile). ssd_dir: the folder on the drive to use; tierlm keeps a copy of those weights there
    // (<model>.tierlm-ssd, written once). Empty: they are read from the model's own file, where it is.
    std::string ssd_dir;
    // 1 always (as much as ram_budget_mb allows: the tests), 0 never (everything left goes in RAM, as before), -1 when
    // RAM is too small (the default).
    int ssd = -1;
    // Most RAM for weights; 0 = what's free less a margin for Windows and other programs (1.5 GB, or 12% of RAM if more).
    long long ram_budget_mb = 0;
};

// The SSD tier's read-ahead for one layer: its expert tensors on the drive (see prefetch_experts in model.cpp).
struct SsdPrefetch { const ggml_tensor* parts[4] = {}; long long n_expert = 0; };

// Where the time goes, accumulated over every eval_batch call (for profiling).
struct EvalStats {
    int calls = 0, tokens = 0;
    double build_ms = 0, alloc_ms = 0, input_ms = 0, compute_ms = 0, output_ms = 0;
    int splits = 0, copies = 0;                  // of the last graph: GPU<->CPU hand-offs and tensor copies
};

// Where the tier loader put things.
struct Placement {
    uint64_t gpu_weights = 0, cpu_weights = 0;   // bytes of weights in VRAM and in system RAM
    uint64_t pinned_weights = 0;                 // of cpu_weights, in pinned memory (see load)
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
    uint64_t ssd_weights = 0;                    // bytes of weights read from the drive as needed (the SSD tier)
    int ssd_layers = 0;                          // layers with weights there (expert layers, in MoE models)
    bool ssd_ngram = false;                      // Flash-Next's n-gram table is there
    std::string ssd_file;                        // the file they're read from
    int cold_mapped_layers = 0;                  // split layers whose RAM experts are read from the model file as used
    // The weights held in RAM (plus 1 GB to work in) against the RAM that was free when loading. ram_need more than
    // ram_free: other programs hold RAM this model counts on, Windows will page, and replies run several times slower
    // (Flash-Next on a 64 GB PC: 18 tok/s that way, 74 after a restart).
    uint64_t ram_need = 0, ram_free = 0;
    bool ram_short() const { return ram_free > 0 && ram_need > ram_free; }
    uint64_t compute_gpu = 0;                    // the first GPU's working memory (compute buffers), bytes
    int vram_reserve_mb = 0;                     // VRAM kept free for it and the driver at this load
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
    static std::unique_ptr<Model> load_once(const std::string& path, const LoadOptions& opt);
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
        std::vector<float> hidden;                // n_tok x hidden_width() hidden states (if asked for), for the MTP block
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
    // Hybrid models (DeltaNet): their running memory can't be rewound, so truncate starts the conversation over...
    // unless there is a checkpoint (an anchor) to go back to: a copy of the running memories as they were at some
    // position. truncate(n) goes back to the last one at or before n, so a message that repeats the conversation up
    // to there carries on from it instead of reading everything again (the reply as stored in the history often
    // differs a little from what was generated, e.g. without its empty thinking block; a coding agent rewrites
    // earlier turns). Several are kept (the cache's keys and values up to each are still there, so an anchor is
    // only the running memories: about 110 MB each for Flash-Next), at most 12 and 1.5 GB (TIERLM_ANCHORS,
    // TIERLM_ANCHORS_MB); when there are too many, the one closest to its neighbours goes.
    void checkpoint();
    int checkpoint_pos() const { return ckpts_.empty() ? -1 : ckpts_.back().n_past; }   // the last one (-1: none)
    int checkpoint_before(int n) const {                          // where truncate(n) would go back to (-1: the start)
        int best = -1;
        for (const Snapshot& c : ckpts_) if (c.n_past <= n && c.n_past <= n_past_) best = c.n_past;
        return best;
    }
    int checkpoints() const { return (int)ckpts_.size(); }

    // A conversation's memory, copied out to system RAM so another conversation can use the cache meanwhile and
    // this one can come back without being read again: the keys and values of its n_past positions in every
    // attention layer (and the MTP draft block), and the running memories of DeltaNet layers.
    struct Snapshot { int n_past = 0; std::vector<std::vector<uint8_t>> parts; size_t bytes = 0;
                      std::vector<std::array<int, 3>> rope;     // (images) each position's 3-part position
                      std::vector<int32_t> tok;                 // (Flash-Next) the tokens, for its n-grams
                      std::vector<std::vector<float>> qsa; };   // (Flash-Next) each indexed layer's indexer keys
    Snapshot save() const;
    // Live hot experts (LoadOptions::live_hot): how well VRAM is serving the router's picks.
    struct LiveStats { long long picks = 0, hits = 0, swaps = 0; double swap_ms = 0; };
    const LiveStats& live_stats() const { return live_stats_; }
    void reset_live_stats() { live_stats_ = LiveStats(); }
    bool live_hot() const { return !live_.empty(); }
    // Find the fastest settings: the VRAM experts held where they are (frozen, a live load serves as the fixed set the
    // busiest experts gave it), so the fixed and the following set are both measured from one load
    void set_live_frozen(bool f) { live_frozen_ = f; }
    bool live_frozen() const { return live_frozen_; }
    // Guessing ahead with the model itself, cut down to what's in VRAM: while on, each split expert layer uses only
    // the router's picks that are in VRAM (their weights scaled up to the picks' whole weight) and skips the RAM part,
    // so a step runs on the GPU alone. An approximation of the model, for guesses that the whole model then checks.
    void set_hot_only(bool on) { hot_only_ = on; }
    // Guessing with the model itself writes its own memory: draft_mark() remembers where the conversation is (and
    // copies the running memories of DeltaNet layers, which can't be rewound otherwise, next to them in the same
    // memory), draft_rewind() puts it back there. False if this model has memory that can't be put back this way.
    bool draft_mark();
    void draft_rewind();
    bool draft_ok() const;
    bool hot_only() const { return hot_only_; }
    bool has_split_experts() const;
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
    bool flash_attn() const { return flash_attn_ && !hp_.ink; }   // (Inkling 2's attention has a bias per head and pair: worked out in the open)
    const char* kv_type() const;      // "f16", "f32", "q8_0", "q5_0" or "q4_0"
    int batch() const { return batch_; }
    // The most tokens a check of guesses can have (the token before them and the guesses) without taking more
    // working memory than is kept free: at least 5; up to 9 where a prompt batch's memory is larger anyway.
    int check_width() const { return check_width_; }
    // The most prompt tokens to read in one step that starts at position n_past. Attention's working memory goes
    // with (tokens in the step) x (the cache span they attend over): the mask alone is 2 bytes for each pair, 2.1 GB
    // for 4,096 tokens at 262,144. Kept free on the GPU for the worst step, that memory held no weights: Flash-Next
    // on an RTX 5090 had 5,786 experts in VRAM at a 256K context against 8,930 at 64K, and replies checked their
    // guesses in 105 ms instead of 44 whatever the conversation's real length. So past batch_span_ positions (65,536)
    // a step takes fewer tokens, the product staying the same: the worst step is then the same at any context length.
    int batch_at(int n_past) const;
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
        bool kept = false;                       // belongs to a kept plan (see Plan): not freed after its step
        ggml_tensor *img = nullptr, *keep = nullptr;   // images: image words [n_embd, n_tok], 1 for text tokens [1, n_tok]
        bool mrope = false;                      // pos holds 4 x n_tok 3-part positions
        ggml_tensor *hidden = nullptr;           // final-norm hidden states of every position (for the MTP head)
        ggml_tensor *mask_swa = nullptr;         // causal mask limited to the sliding window (models that use one)
        ggml_tensor *cells_swa = nullptr;        // with rings (swa_R_): where each token's key and value go in them
        ggml_tensor *rope_ff = nullptr;          // YaRN frequency factors (Hparams::rope_ff), as input
        ggml_tensor *h_in = nullptr;             // MTP head: the target model's hidden states, as input
        ggml_tensor *cells = nullptr;            // where each token's key and value go in the cache
        ggml_tensor *zero_id = nullptr;          // Mamba's scan: which state to start from (always the one given: 0)
        ggml_tensor *out_idx = nullptr;          // several conversations: the tokens whose logits are wanted
        ggml_tensor *ngram = nullptr;            // Flash-Next: each token's n-gram rows in the PLE table [heads * n_tok]
        std::vector<std::pair<int, ggml_tensor*>> qsa_k;   // QSA: per indexed layer, this batch's indexer keys
        // QSA on the GPU: the blocks a step touches (their tokens' rows of raw keys, the row each one's key is kept
        // in, their first tokens' positions); each token's complete blocks; the blocks the mask covers
        ggml_tensor *qsa_crow = nullptr, *qsa_clast = nullptr, *qsa_cpos = nullptr, *qsa_nb = nullptr;
        int64_t qsa_nbt = 0;
        int n_tok = 0;
        // Per layer (DeltaNet layers only): for dropping the last d tokens, rollback[layer][d] = {memory, conv inputs}
        std::map<int, std::vector<std::pair<ggml_tensor*, ggml_tensor*>>> rollback;
        // Per PLE layer (Flash-Next): the convolution's kept inputs as they were d tokens back, ple_rollback[layer][d]
        std::map<int, std::vector<ggml_tensor*>> ple_rollback;
        // Inkling 2: each token's log scale [1, 1, n_tok]; the relative-bias row per cache cell and token [n_kv, n_tok]
        // (full-attention layers; sliding layers: their cells or ring slots); per layer and pass (key: the state
        // tensor) the convolutions' inputs as they were d tokens back, ink_rollback[state][d]
        ggml_tensor *ink_tau = nullptr, *ink_rel = nullptr, *ink_rel_swa = nullptr;
        ggml_tensor *ink_capv = nullptr;         // (the experts' softcaps: 1 / cap for each row of gate and up, as input)
        std::map<ggml_tensor*, std::vector<ggml_tensor*>> ink_rollback;
        std::vector<ggml_tensor*> orders;        // each layer's chosen experts (when routing is wanted)
        std::vector<int> order_layers;           // the layer each of orders belongs to
        ggml_tensor* routing = nullptr;          // all of them joined: [n_used, n_tok * n_layer], one read
        std::vector<std::pair<ggml_tensor*, int>> owner;   // every op tensor of each layer, with its layer
        std::vector<ggml_tensor*> cpu_ops;       // small ops that must run on the CPU (hot-expert bookkeeping)
        std::vector<std::pair<ggml_tensor*, int>> stream_marks;   // streamed prompt batch: each split layer's RAM experts' result, its turn
        // inputs made with the graph (DeepSeek V4: which ring rows and entries each step reads and writes, masks)
        std::vector<std::pair<ggml_tensor*, std::vector<int32_t>>> in_i32;
        std::vector<std::pair<ggml_tensor*, std::vector<float>>> in_f32;
    };
    // A batch's tokens, conversation by conversation (contiguous): n tokens of conversation seq.
    struct SeqGroup { int seq, n; };
    // n_kv_end: the cache positions attention covers (all used ones), at least n_tok.
    Graph build(int n_tok, int n_kv_end, bool want_routing, int n_logits = 1, bool want_hidden = false, int snapshots = 1,
                const std::vector<SeqGroup>* groups = nullptr, std::vector<uint8_t>* mem = nullptr);
    struct Layer;
    ggml_tensor* attn_block(ggml_context* c, Graph& g, const Layer& L, ggml_tensor* h, int n_tok, int n_past);
    ggml_tensor* attn_mla(ggml_context* c, Graph& g, const Layer& L, ggml_tensor* h, int n_tok, int n_past);
    ggml_tensor* ffn_block(ggml_context* c, Graph& g, const Layer& L, ggml_tensor* h, int n_tok, bool want_routing,
                           ggml_tensor* router_in = nullptr);   // (Gemma 4: the router reads its own input)
    ggml_tensor* attn_gemma4(ggml_context* c, Graph& g, const Layer& L, ggml_tensor* h, int n_tok, int n_past);
    // Inkling 2: every layer (and loop pass) from the normed embedding to the last depth mixer's output
    ggml_tensor* ink_layers(ggml_context* c, Graph& g, ggml_tensor* x, int n_tok, int n_kv, const std::vector<SeqGroup>& grp,
                            int snapshots, bool want_routing, const std::function<void(int)>& tag);
    void pin_cpu_layers(const Graph& g, ggml_backend_sched* sc = nullptr);        // keep every op of a RAM layer on the CPU
    std::vector<float> eval_batch(const int32_t* ids, int n_tok, int n_threads, std::vector<int32_t>* routing);
    void release_pending();
    // One graph: n_tok tokens of conversation 0 (groups null), or of the conversations in groups.
    EvalOut step(const int32_t* ids, int n_tok, int n_threads, int n_logits, bool want_hidden, int snapshots,
                 std::vector<int32_t>* routing, const std::vector<SeqGroup>* groups);
    Graph build_mtp(int n_tok, int n_past, std::vector<uint8_t>* mem = nullptr, bool images = false);
    void* pending_ = nullptr;                    // Graph of the last undoable batch, until accept()
    // Live hot experts: per split layer, which expert each VRAM slot holds and how much each is wanted.
    struct LivePart { ggml_tensor *hot, *cold; };   // an expert tensor's VRAM part and its RAM part (the rest)
    struct LiveLayer {
        int layer = 0, H = 0;
        std::vector<int> slot_of, in_slot;       // expert -> VRAM slot (-1: not there); slot -> expert
        std::vector<int> cold_of;                // expert -> RAM slot (-1: in VRAM)
        std::vector<float> score;                // learned use, then recent picks (fading)
        std::vector<LivePart> parts;
        ggml_tensor *map_hot = nullptr, *map_cold = nullptr, *mask = nullptr;
        ggml_tensor *map_hot_g = nullptr, *hot_bias_g = nullptr;   // (their VRAM copies, see Layer)
    };
    std::vector<LiveLayer> live_;
    mutable LiveStats live_stats_;
    bool live_frozen_ = false;
    void adapt_hot(const std::vector<int32_t>& routing, int n_tok);
    std::vector<char> swap_buf_;                 // (adapt_hot: the incoming experts on their way)
    double live_step_ms_ = 0, live_ms_per_mb_ = 0;   // (adapt_hot: a step's time, and moving's cost per MB, both averaged)
    std::chrono::steady_clock::time_point live_last_{};
    size_t live_credit_ = 0;                     // (adapt_hot: moving budget left unused by earlier steps)
    struct Layer {
        ggml_tensor *attn_norm = nullptr, *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
        ggml_tensor *bq = nullptr, *bk = nullptr, *bv = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        ggml_tensor *bo = nullptr, *sinks = nullptr;                          // gpt-oss: output bias, attention sinks
        ggml_tensor *router_b = nullptr, *gate_xb = nullptr, *up_xb = nullptr, *down_xb = nullptr;   // gpt-oss: biases
        bool rope = true;                        // rotates positions (see Hparams::rope_layer)
        int index = 0;                           // the layer's number
        const SsdPrefetch* ssd = nullptr;        // its experts are on the drive (the SSD tier): read them ahead
        // MLA (deepseek2): query compression (or plain wq), the compressed key/value projection with its norm, and
        // per head: k_b [nope, kv_lora] (a query's unrotated part into the compressed space), v_b [kv_lora, v_dim].
        ggml_tensor *wq_a = nullptr, *q_a_norm = nullptr, *wq_b = nullptr, *wkv_a = nullptr, *kv_a_norm = nullptr;
        ggml_tensor *wk_b = nullptr, *wv_b = nullptr;
        ggml_tensor* exp_probs_b = nullptr;      // router bias used only to choose experts (DeepSeek V3)
        // Flash-Next: the hyper-connection mixers before attention and before the experts (norm over each copy of
        // the width, low-rank mixing weights, and how much of the block's output each copy gets)...
        ggml_tensor *hc_a_norm = nullptr, *hc_a_down = nullptr, *hc_a_up = nullptr, *hc_a_inj = nullptr;
        ggml_tensor *hc_f_norm = nullptr, *hc_f_down = nullptr, *hc_f_up = nullptr, *hc_f_inj = nullptr;
        // ...and the per-layer embedding (on the layers that have one), with its convolution's saved inputs per
        // conversation ([hc * n_embd, ple_state_len] each)
        ggml_tensor *ple_key = nullptr, *ple_value = nullptr, *ple_norm_key = nullptr, *ple_norm_query = nullptr;
        ggml_tensor *ple_norm_conv = nullptr, *ple_conv = nullptr;
        std::vector<ggml_tensor*> ple_states;
        ggml_tensor *idx_q = nullptr, *idx_k = nullptr, *idx_q_norm = nullptr, *idx_k_norm = nullptr;   // QSA indexer
        // QSA: every position's raw indexer key, by cache cell [qsa_dim, ctx] f16; a whole block's last row holds the
        // block's key instead (see qsa_mask_gpu)
        ggml_tensor *qsa_kc = nullptr;
        ggml_tensor *attn_post_norm = nullptr, *ffn_post_norm = nullptr;   // Gemma: norms on the blocks' outputs
        ggml_tensor *ffn_pre_norm2 = nullptr, *ffn_post_norm1 = nullptr, *ffn_post_norm2 = nullptr;   // Gemma 4 MoE: the two paths' norms
        ggml_tensor *router_scale = nullptr, *exp_scale = nullptr;   // Gemma 4: the router input's scale, each expert's output scale
        ggml_tensor *out_scale = nullptr;        // Gemma 4: the layer's output scaled by this
        ggml_tensor *ple_gate = nullptr, *ple_proj = nullptr, *ple_post = nullptr;   // Gemma 4 E2B/E4B: per-layer input
        int qsa = -1;                            // its entry in qsa_
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
        ggml_tensor *gate_xbh = nullptr, *up_xbh = nullptr, *down_xbh = nullptr;   // the biases' VRAM copies (gpt-oss; whole)
        ggml_tensor *map_hot = nullptr, *map_cold = nullptr, *hot_mask = nullptr;   // [1, n_expert]
        // set_hot_only: the same in VRAM, so a step with only the VRAM experts doesn't visit the CPU: map_hot, and
        // a bias on the router's choice (0 in VRAM, -1e9 not), so the picks are the best of those in VRAM.
        ggml_tensor *map_hot_g = nullptr, *hot_bias_g = nullptr;
        bool cold_mapped = false;                // the RAM part is the file's whole expert tensor (see load: RAM short)
        int stream_turn = -1;                    // streamed prompt batches: this layer's turn among those streamed (see Stream)
        ggml_tensor *gate_xs = nullptr, *up_xs = nullptr, *down_xs = nullptr, *gate_up_xs = nullptr;   // its RAM part's VRAM slot
        ggml_tensor *up_s = nullptr, *gate_s = nullptr, *down_s = nullptr, *gate_inp_s = nullptr;   // shared expert
        ggml_tensor *k_cache = nullptr, *v_cache = nullptr;   // [head_dim, n_head_kv, max_ctx]
        // Hybrid models: Gated DeltaNet layer
        bool recurrent = false;
        ggml_tensor *wqkv = nullptr, *wgate = nullptr, *ssm_conv1d = nullptr, *ssm_dt = nullptr, *ssm_a = nullptr;
        ggml_tensor *ssm_beta = nullptr, *ssm_alpha = nullptr, *ssm_norm = nullptr, *ssm_out = nullptr;
        ggml_tensor *ssm_ba = nullptr, *ssm_in = nullptr;    // Qwen3-Next: fused beta/alpha, and older fused q/k/v/z
        ggml_tensor *conv_state = nullptr;       // [ssm_conv - 1, conv_dim]: the last few inputs
        // Nemotron: what the block is (Hparams::Block); Mamba-2's convolution bias, D and the scales NVFP4 files
        // keep beside in/out; LatentMoE's projections into and out of the experts' width
        int kind = 0;
        ggml_tensor *ssm_conv_b = nullptr, *ssm_d = nullptr, *ssm_in_s = nullptr, *ssm_out_s = nullptr;
        ggml_tensor *lat_down = nullptr, *lat_up = nullptr;
        // MTP draft block only
        ggml_tensor *eh_proj = nullptr, *enorm = nullptr, *hnorm = nullptr, *mtp_embd = nullptr, *head_norm = nullptr, *head = nullptr;
        ggml_tensor *hc_head_norm = nullptr, *hc_head_down = nullptr, *hc_head_up = nullptr;   // (Flash-Next's draft block: its own last mixer)
        // Inkling 2 (see Hparams::ink): the output gate, query scale, sink scores, relative states (projection, scale)
        // and their bias table [rel_dim, extent]; the convolutions' taps [conv, channels] (keys, values, attention
        // output, experts' output) and, in full-attention layers, the per-token taps' projections and the separate
        // query/key/value input norms; the depth mixers' queries (attention input: 1 or 3, experts' input: 1); in loop
        // layers the 4 message queries and the q/k/v/ffn gates' down, gain and forget projections.
        ggml_tensor *ink_gate = nullptr, *ink_qclip = nullptr, *ink_sink = nullptr, *ink_rel = nullptr, *ink_relb = nullptr;
        ggml_tensor *ink_relclip = nullptr, *ink_kconv = nullptr, *ink_vconv = nullptr, *ink_aconv = nullptr, *ink_fconv = nullptr;
        ggml_tensor *ink_kconvp = nullptr, *ink_vconvp = nullptr, *ink_qn = nullptr, *ink_kn = nullptr, *ink_vn = nullptr;
        ggml_tensor *ink_ares = nullptr, *ink_fres = nullptr, *ink_msgq = nullptr;
        // experts stored with a scale each beside their weights (NVFP4 files): the gate+up, gate and up matrices'
        // (the down matrix's is exp_scale)
        ggml_tensor *x_scale_gu = nullptr, *x_scale_gate = nullptr, *x_scale_up = nullptr;
        ggml_tensor *ink_down[4] = {}, *ink_gain[4] = {}, *ink_forget[4] = {};
        std::vector<ggml_tensor*> ink_states;    // per conversation: the convolutions' last inputs [conv - 1, channels]
        ggml_tensor *ssm_state_t = nullptr;      // [S, S, H_v]: the running memory
        std::vector<ggml_tensor*> conv_states, ssm_states;   // per conversation ([0]: conv_state, ssm_state_t)
        // DeepSeek V4: the stream mixers (fn, base, scale; [0] before attention, [1] before the experts), attention
        // (q down/norm/up, the one key/value head and its norm, grouped output down and up), the compressor (kv,
        // gate, its positional bias, norm) and the indexer (query up, head weights, and its own compressor); the
        // first layers' token -> experts table. Their memory: the last tokens' keys (a ring by position), the
        // compressed entries, and the last tokens' compressor inputs (rings): see Model::d4_ring.
        ggml_tensor *d4_hc_fn[2] = {}, *d4_hc_base[2] = {}, *d4_hc_scale[2] = {};
        ggml_tensor *d4_q_a = nullptr, *d4_q_a_norm = nullptr, *d4_q_b = nullptr, *d4_kv = nullptr, *d4_kv_norm = nullptr;
        ggml_tensor *d4_o_a = nullptr, *d4_o_b = nullptr;
        ggml_tensor *d4_c_kv = nullptr, *d4_c_gate = nullptr, *d4_c_ape = nullptr, *d4_c_norm = nullptr;
        ggml_tensor *d4_i_q_b = nullptr, *d4_i_proj = nullptr;
        ggml_tensor *d4_ic_kv = nullptr, *d4_ic_gate = nullptr, *d4_ic_ape = nullptr, *d4_ic_norm = nullptr;
        ggml_tensor *d4_tid2eid = nullptr;
        ggml_tensor *d4_win = nullptr;                       // [head_dim, ring]: the last tokens' keys (= values)
        ggml_tensor *d4_ce = nullptr, *d4_ie = nullptr;      // [head_dim | index dim, entries]: compressed entries
        ggml_tensor *d4_c_rkv = nullptr, *d4_c_rgate = nullptr, *d4_i_rkv = nullptr, *d4_i_rgate = nullptr;   // [width, ring]
    };
    Hparams hp_;
    ggml_backend* main_ = nullptr;               // device holding weights + cache (the first GPU)
    std::vector<ggml_backend*> gpus_;            // every GPU in use, main_ first
    ggml_backend* cpu_ = nullptr;                // fallback for ops the device can't run
    ggml_backend_sched* sched_ = nullptr;
    ggml_context* wctx_ = nullptr;               // weight tensor descriptors
    ggml_context* kvctx_ = nullptr;              // KV cache tensor descriptors
    std::vector<ggml_backend_buffer*> wbufs_;    // weight memory: one buffer in VRAM, one in system RAM
    std::vector<SsdPrefetch> ssd_pf_;            // per layer (sized at load, so Layer::ssd pointers stay valid)
    std::vector<std::unique_ptr<MappedFile>> maps_;   // the SSD tier's files (outlive wbufs_, freed in ~Model)
    size_t kept_ram_ = 0;                         // RAM weights held in RAM for this model (keep_in_ram)
    Placement place_;
    EvalStats stats_;
    std::vector<ggml_backend_buffer*> kvbufs_;   // KV cache memory: VRAM part and RAM part
    std::vector<ggml_backend_buffer_type*> kv_bufts_;   // where each part of the cache goes: per GPU, then RAM
    void make_kv(int ctx, int n_seq);
    void free_kv();
    // Scratch for building graphs, one per kind (one token; several tokens; the MTP draft block): the GPU keeps a
    // recorded replay (CUDA graph) per graph, found by where it was built, so each kind keeps its own.
    // Each size of a few-token step (speculative checks, the draft block's steps) gets its own, too: the replay is
    // kept only while the graph built there stays the same, and checks of different sizes take turns.
    std::map<std::pair<int, int>, std::vector<uint8_t>> graph_mem_;
    size_t graph_mem_size_ = 0;
    std::vector<uint8_t>& graph_mem(int kind, int n_tok) {
        std::vector<uint8_t>& v = graph_mem_[{kind, n_tok <= 16 ? n_tok : 0}];
        if (v.size() < graph_mem_size_) v.resize(graph_mem_size_);
        return v;
    }
    ggml_tensor *tok_embd_ = nullptr, *out_norm_ = nullptr, *output_ = nullptr, *rope_freqs_ = nullptr;
    ggml_tensor *g4_ple_tok_ = nullptr, *g4_ple_proj_ = nullptr, *g4_ple_norm_ = nullptr;   // Gemma 4 E2B/E4B: per-layer inputs
    ggml_tensor *tok_norm_ = nullptr, *ink_out_res_ = nullptr;   // Inkling 2: the embedding's norm, the last depth mixer's query
    ggml_tensor *ink_res_q_ = nullptr, *ink_kv_q_ = nullptr;     // (the earlier layout: all loop layers' message queries, see ink_layers)
    // Inkling 2's loop: the caches of the loop layers' later passes, pass p >= 1 of layer i at
    // (p - 1) * (loop_hi - loop_lo) + i - loop_lo. Each holds only its own cache (index: the layer it repeats).
    std::vector<Layer> ink_slots_;
    Layer& ink_slot(int li, int pass) { return pass == 0 ? layers_[li] : ink_slots_[(size_t)(pass - 1) * hp_.ink_loop_n() + li - hp_.ink_loop_lo]; }
    ggml_tensor *ple_embd_ = nullptr;            // Flash-Next: the PLE n-gram table [ple_dim, rows]
    ggml_tensor *out_hc_norm_ = nullptr, *out_hc_down_ = nullptr, *out_hc_up_ = nullptr;   // the last stream mixer
    ggml_tensor *d4_out_fn_ = nullptr, *d4_out_base_ = nullptr, *d4_out_scale_ = nullptr;  // DeepSeek V4: the last stream mixer
    // DeepSeek V4's rings: positions are kept in slot pos % size, with room for d4_slack positions more than the
    // window or compressor needs, so a step of up to d4_slack tokens can be undone (speculative decoding) and the
    // conversation cut back that far; further back it is read again (truncate).
    static constexpr int d4_slack = 64;
    int d4_ring(int need) const { return need + d4_slack; }
    // Sliding-window layers (Gemma 3 and 4, gpt-oss) keep only a ring of swa_R_ positions per conversation instead
    // of the whole context: token index q of conversation s goes to slot s * swa_R_ + q % swa_R_. The ring holds the
    // window, a whole step (kRingStep tokens, the most a step writes: batch_ is capped to it) and kRingSlack more, so
    // a step never writes over what one of its own tokens reads, and a conversation can be cut back up to kRingSlack
    // tokens without reading anything again (further: read again from the start, see free_cells_from). 0: off (the
    // context isn't longer than the rings would be, or TIERLM_SWA_RING=0).
    static constexpr int kRingStep = 4096, kRingSlack = 1024;
    int swa_R_ = 0;
    int ring_extra_ = kRingStep + kRingSlack;   // swa_R_ = n_swa + this
    int swa_kv_build_ = 0;                       // the ring span the next build covers (0: all of it)
    bool ringed(const Layer& L) const;
    Graph build_dsv4(int n_tok, int n_past, bool want_routing, int n_logits, bool want_hidden, std::vector<uint8_t>* mem);
    std::vector<std::vector<int32_t>> seq_tok_;  // Flash-Next: each conversation's tokens (for the n-grams)
    // QSA (Flash-Next's sparse attention): each indexed layer's raw indexer keys, per conversation, by position
    // (qsa_dim numbers each), and that layer's norm weights; kept in RAM, read by the mask-making op.
public:
    struct QsaCall { const Model* m; int i; };
    struct QsaLayer { int layer = -1, ratio = 0; std::vector<float> qn, kn; std::vector<std::vector<float>> keys;
                      std::vector<std::vector<float>> blocks; };   // per conversation: finished blocks' keys, normed and rotated
    struct QsaBatch {                           // the step being computed: each token's conversation, position, rotation
        std::vector<int> seq, pos; std::vector<std::array<int, 3>> rope; std::map<std::pair<int, int>, int> at;   // (seq, pos) -> token
    };
    const QsaLayer& qsa_layer(int i) const { return qsa_[i]; }
    void qsa_block_key(const QsaLayer& Q, const float* const* raw, const std::array<int, 3>& pos, float* out) const;
    void qsa_mask(const QsaLayer& Q, const ggml_tensor* mask, const ggml_tensor* q, const ggml_tensor* k, ggml_tensor* dst, int ith, int nth) const;
private:
    std::vector<QsaLayer> qsa_;
    std::vector<QsaCall> qsa_calls_;            // the mask-making op's userdata, one per indexed layer
    QsaBatch qsa_batch_;
    bool qsa_long_ = false;                      // this step has a query past the budget: build the sparse masks
    bool qsa_kc_ok_ = true;                      // qsa_kc holds every position's key (false after a restore until qsa_upload)
    // qsa_kc holds the key of every whole block of conversation 0 in the block's last row. The GPU's mask-making steps
    // keep it so; after any other step (within the budget, with an image, several conversations) or a restore the
    // keys are sent again from the host's copy before the next one (qsa_upload_blocks).
    bool qsa_bk_ok_ = false;
    void qsa_upload_blocks();
    void qsa_make_blocks();
    bool qsa_gpu_ = false;                       // ... made by the graph (on the layer's device), not on the CPU (see qsa_mask_gpu)
    ggml_tensor* qsa_mask_gpu(ggml_context* c, Graph& g, const Layer& L, ggml_tensor* kc, ggml_tensor* qr, ggml_tensor* mask, int n_tok);
    void qsa_upload();                           // the CPU's copy of conversation 0's indexer keys into qsa_kc (after a restore)
    std::vector<Layer> layers_;
    std::unique_ptr<Layer> mtp_;                 // the MTP draft block, when loaded
    std::vector<bool> layer_kv_gpu_;             // per layer: KV cache in VRAM
    std::vector<int> layer_dev_;                 // per layer: the GPU (index into gpus_) holding its KV cache, or -1 for RAM
    struct { int graphs = 0; double tokens = 0, prep = 0, build = 0, alloc = 0, compute = 0, out = 0; } prof_;   // TIERLM_PROFILE
    // Attention covers the cache in steps of 256 positions (the extra ones are masked out), so the graph keeps the
    // same shapes from one token to the next and the GPU can replay it (CUDA graphs) instead of launching each op.
    int kv_span(int n) const { const int s = (n + 255) / 256 * 256; return s < max_ctx_ ? s : (n > max_ctx_ ? n : max_ctx_); }
    // a reply's one-token graph, kept to run again (see step)
    struct ReuseKey {
        int n_kv = 0; bool routing = false; int n_logits = 0; bool qsa_long = false; bool hidden = false;
        int n_tok = 1, snapshots = 1; bool hot_only = false, qsa_gpu = false; int n_kv_swa = 0;
        bool operator==(const ReuseKey& o) const {
            return n_kv == o.n_kv && n_kv_swa == o.n_kv_swa && routing == o.routing && n_logits == o.n_logits && qsa_long == o.qsa_long && hidden == o.hidden &&
                   n_tok == o.n_tok && snapshots == o.snapshots && hot_only == o.hot_only && qsa_gpu == o.qsa_gpu;
        }
    };
    Graph reuse_;
    ReuseKey reuse_key_;
    void drop_reuse();
    // Guessing ahead runs three kinds of graph in turn: the draft block for one token (once per guess), the model's
    // check of the guesses, and the draft block catching up. One scheduler holds one placed graph, so each of them
    // used to be built and placed again every time: about 4 ms of a 45 ms check on an RTX 5090, and the same again
    // over the draft block's steps. The check and the draft block's one-token step each get a scheduler of their
    // own (with its own working memory), so their graphs stay placed and are run again, like a reply's one-token
    // graph above. Only for models with a draft block, and when the check's memory is a small part of the card.
    // TIERLM_PLANS=0: as before.
    // Since 0.9.4 the check's graph is kept on the main scheduler instead (check_main_): that one holds the working
    // memory of a whole prompt batch, idle while a reply is written, so the check needs none of its own (695 MB for
    // Flash-Next on an RTX 5090: 330 more experts in VRAM) and a wider check costs no memory either. The draft
    // block's catching-up step (2 to 9 tokens), which used the main scheduler between checks, has its own small one.
    struct Plan { Graph g; ReuseKey key; ggml_backend_sched* sched = nullptr; std::vector<uint8_t> mem; };
    Plan check_, mtp1_, mtpk_;
    // A check's graph is built in memory of its own size's: built again (after a prompt step, or a check of another
    // size, took the scheduler) its tensors are where they were, and the GPU engine, which keeps a recorded run of each
    // graph it has seen twice (by its first tensor and what it holds), runs the recording again instead of warming up
    // and recording anew. Sharing one memory, a check of another size cost the next checks 3 to 13 ms each.
    std::vector<uint8_t> check_mems_[10];
    std::vector<uint8_t>& check_mem(int n_tok) {
        std::vector<uint8_t>& v = check_mems_[std::min(std::max(n_tok, 0), 9)];
        if (v.size() < graph_mem_size_) v.resize(graph_mem_size_);
        return v;
    }
    int mtpk_w_ = 9; bool check_main_ = false;                                      // the check's graph stays placed on sched_
    int check_width_ = 5;                                          // tokens in a check that sched_'s memory holds (see check_width)
    void drop_plan(Plan& p);
    void drop_plans();                                             // reuse_ and both plans (the cache changes size)
    bool overlap_ = true;                                          // CPU and GPU at once on split expert layers
    bool hot_only_ = false;                                        // set_hot_only: only the experts in VRAM
    int draft_pos_ = -1;                                           // draft_mark: where the conversation was
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> draft_keep_; // (a running memory, its copy)
    std::vector<ggml_context*> draft_ctx_;
    std::vector<struct ggml_backend_buffer*> draft_buf_;
    float cold_skip_ = 0.0f;                                       // TIERLM_COLD_SKIP (see the split expert layers)
    std::vector<Snapshot> ckpts_;                                  // (see checkpoint) in order of position
    int n_threads_last_ = 0;                                       // (for the profile)
    void profile_step(int n_tok, double prep_ms, double compute_ms, double out_ms, double build_ms = 0, double alloc_ms = 0);
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
    std::vector<ggml_tensor*> ink_states_of(int s) const;   // Inkling 2: conversation s's convolution inputs, every layer and pass
    void clear_state(int s);
    void fill_ink_inputs(Graph& g, int n_kv, const std::vector<int>& tok_seq, const std::vector<int>& tok_pos);                     // DeltaNet memories of conversation s back to zero
    uint64_t weights_bytes_ = 0, kv_bytes_ = 0;
    void* dev_ = nullptr;                        // ggml_backend_dev_t of the main device
    bool flash_attn_ = true, kv_f32_ = false;
    int kv_t_ = 1;                               // ggml_type of the KV cache (GGML_TYPE_F16)
    int batch_ = 256, n_expert_used_file_ = 0;
    int batch_span_ = 65536;                     // (batch_at: TIERLM_BATCH_SPAN, 0: batches never shrink)
    int batch_gran_ = 256;                       // (batch_at: the smallest step, and what steps shrink by)
    int cold_offload_min_ = 32;                  // (see load: when the GPU takes over the RAM experts' work in a batch)
    int stream_min_ = 512;                       // ... and from which a streamed layer's are (see setup_stream)
    // Streamed prompt batches (see setup_stream): the split layers' RAM experts are copied into two VRAM slots on a
    // second GPU queue, the next layer's while this one computes, instead of by the scheduler before each layer.
    struct Stream {
        ggml_backend* copy = nullptr;            // the second queue (another CUDA engine on the same card)
        ggml_backend_buffer* buf = nullptr;      // the two slots
        ggml_context* ctx = nullptr;             // the slot tensors, one set per streamed layer (two layers share a slot)
        ggml_backend_event* ev[2] = {nullptr, nullptr};
        std::vector<int> layers;                 // the streamed layers, in order (their turns)
        size_t bytes = 0;                        // both slots
        std::vector<std::vector<ggml_tensor*>> pads;   // per turn: the CUDA row padding after each slot tensor (zeroed after each copy)
        ggml_backend_buffer* zeros = nullptr;    // as many zeros as the biggest padding, pinned (a copy from ordinary memory
                                                 // waits for the queue: the next layer's work then waited for the copies)
    };
    Stream stream_;
    bool stream_on_ = false;                     // this batch is streamed (built with the slots)
    void setup_stream();
    bool stream_ready(int n_tok) const;          // a batch this big is streamed
    bool reserve_worst(bool probe = false);      // compute memory for the largest steps (see batch_at)
    void stream_send(int turn);                  // copy that turn's layer into its slot (on the second queue), then mark it
};
using OlmoeModel = Model;

}
