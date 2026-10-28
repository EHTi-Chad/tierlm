# How tierlm works

## The idea

Large language models normally have to fit in GPU memory (VRAM) to run fast. **Mixture-of-experts (MoE)**
models change that. Each layer contains many small "expert" networks, and a **router** picks only a few of
them for each token. OLMoE-1B-7B, for example, has 64 experts per layer and uses 8.

Measured with `tierlm-inspect` on OLMoE (Q4_K_M, 3.92 GiB):

| Part | Share of the file | Used by every token? |
|---|---|---|
| Experts | 92.6% | No: 8 of 64 per layer |
| Attention | 3.8% | Yes |
| Output layer | 2.0% | Yes |
| Embedding table | 1.4% | One row per token |
| Router + norms | 0.2% | Yes |

So the small, always-used parts can live in fast memory, and the real question is where each of the 1,024
individual experts (64 per layer × 16 layers, about 3.6 MiB each) should go. If the busiest experts sit in
VRAM and the rest in RAM, or even on the SSD, a model far larger than the GPU can still run at usable speed.

| Tier | Example (RTX 5090 workstation) | Approx. read speed | Holds |
|---|---|---|---|
| VRAM | 32 GB GDDR7 | ~1,800 GB/s | Attention, router, norms, hottest experts, KV cache |
| RAM | 64 GB DDR5-6000 | ~90 GB/s | Warm experts, embedding table |
| SSD | NVMe | ~7 GB/s | Cold experts, read only when routed to |

## What is tierlm's own code, and what is borrowed

```
tierlm/
├── src/           tierlm's engine
│   ├── model_map     reads a GGUF header: every tensor's size, type, offset, layer, role
│   ├── planner       splits experts into units, assigns VRAM / RAM / SSD within a budget (dry run)
│   ├── tokenizer     byte-level BPE (GPT-2, Llama 3, Qwen splitting) and SentencePiece, added tokens
│   └── model         one compute graph for every supported architecture, KV cache, flash attention,
│                     batching, device placement
├── tools/         command-line tools and the web server
├── web/           the web page and dashboard (plain HTML/JS, no build step)
├── tests/         reference tests (Python, testing only)
└── third_party/
    ├── ggml/      math kernels, quantized formats, device backends, GGUF reader  (MIT, submodule)
    ├── httplib/   HTTP server                                                    (MIT, single header)
    └── json/      JSON                                                           (MIT, single header)
```

tierlm deliberately does **not** use llama.cpp. ggml supplies the tensor operations, the same way a game
engine relies on DirectX or Vulkan: fast matrix multiplies on quantized weights, rotary embeddings, flash
attention kernels, and backends for CPU, Vulkan and CUDA. Everything that decides *what* to compute and
*where* things live is tierlm's.

## How one token is produced

1. **Tokenizer.** Text is split into words, numbers, punctuation and whitespace (a hand-written version of
   the GPT-2 split pattern, using generated Unicode tables), each piece's bytes are mapped to GPT-2's byte
   alphabet, and BPE merges join them into vocabulary tokens. Added tokens (such as OLMoE's dedicated
   whitespace-run tokens, or `<|user|>` in chat mode) are matched first, longest match wins.
2. **Embedding.** Each token ID becomes a 2,048-number vector.
3. **16 layers**, each:
   - **Attention.** RMSNorm, then Q/K/V projections. OLMoE normalises Q and K across the full width
     (QK-norm), then applies rotary position embeddings (GPT-NeoX style). The new keys and values are written
     into the **KV cache**, and **flash attention** compares the query with every cached key in one fused
     operation, with a causal mask so tokens never see the future.
   - **Router.** RMSNorm, then a small matrix scores all 64 experts; softmax turns scores into weights.
   - **Experts.** The top 8 run (`mul_mat_id` selects them per token): each is a SwiGLU feed-forward network.
     Their outputs are multiplied by their router weights and summed. OLMoE does not renormalise the top-8
     weights.
4. **Output.** A final RMSNorm and the output matrix give a score for each of the 50,304 vocabulary tokens.
   The server samples from these (temperature + top-p); `tierlm-run` takes the highest.

Then the chosen token is fed back in, and the loop repeats. Thanks to the KV cache, each new step processes
only that one token.

The steps above use OLMoE as the example. Other families are variations on the same design, described by a
handful of switches in `Hparams` (src/model.h), so one graph builder runs them all:

| Switch | Llama 2/3, Mistral | Mixtral | Qwen2 | Qwen3 | Qwen3 MoE | OLMoE |
|---|---|---|---|---|---|---|
| Feed-forward | dense | experts | dense | dense | experts | experts |
| RoPE layout | adjacent pairs | adjacent pairs | halves (NeoX) | halves | halves | halves |
| QK-norm | none | none | none | per head | per head | full width |
| Q/K/V biases | no | no | yes | no | no | no |
| Expert weights renormalised | | yes | | | yes | no |
| Tokenizer | Llama 3 BPE / SentencePiece | SentencePiece | Qwen BPE | Qwen BPE | Qwen BPE | GPT-2 BPE |

Grouped-query attention (fewer key/value heads than query heads), a head size independent of the model
width (Qwen3), tied input/output embeddings and Llama 3.1's RoPE frequency factors are handled for all of
them. The chat format (Tulu, ChatML, Llama 3, `[INST]`) is read from the model's own chat template.

**Hybrid models (Qwen3.5 / 3.6 / 3.8, GGUF archs `qwen35` and `qwen35moe`).** Three of every four layers replace attention with
**Gated DeltaNet**: a short causal convolution over the last four inputs, then a per-head running memory (a
128×128 matrix) that decays and is updated a little with every token (ggml's `gated_delta_net` op), followed by a
gated RMSNorm. Only every fourth layer uses full attention, with an output gate, per-head QK-norm and RoPE on the
first quarter of each head. Only those layers need a KV cache, so long conversations cost a quarter of the memory;
the running memories are small, fixed-size, and live with their layer on the GPU or in RAM. They can't be rewound,
so editing an earlier message recomputes the conversation. Draft (multi-token prediction) blocks in the file are
skipped and never loaded. The MoE versions (`qwen35moe`, e.g. Qwen3.6-35B-A3B) replace the feed-forward block
with routed experts (softmax, top 8, renormalised) plus a **shared expert** that runs on every token, scaled by its
own sigmoid gate; files that store each expert's gate and up weights as one tensor (`ffn_gate_up_exps`) are read
as-is, with one matrix multiply for both halves. **Qwen3-Next** (`qwen3next`, e.g. Qwen3-Coder-Next) is
the same math stored differently: its value heads stay grouped by key head (tierlm repeats each key head in
place instead of relying on ggml's tiled broadcast), beta and alpha come as one tensor interleaved per key head
(`ssm_ba`), and older files keep q, k, v and z interleaved per key head in one tensor (`ssm_in`).

## Speculative decoding

When a model doesn't fit in VRAM, generating a token is limited by how fast its weights in RAM can be read: a dense
27B model with 14 GB in RAM manages about 3 tokens/second on a laptop. Speculative decoding gets several tokens
out of each read. Qwen3.5 / 3.6 / 3.8 and Qwen3-Next ship a small **MTP draft layer** (multi-token prediction):
one full-attention block that takes the token just chosen and the model's last hidden state, and predicts the
next token. Unsloth publishes it as a separate `mtp-<model>.gguf`; tierlm loads it from either place.

Each step (`src/speculate.cpp`):

1. **Guess.** The draft layer runs up to `draft_max` times (default 3), feeding its own hidden state forward,
   which is cheap: one layer, usually in VRAM.
2. **Check.** The model evaluates the chosen token and the guesses in **one pass** (`Model::eval_ex`), producing a
   prediction for every position. Its weights are read once for all of them.
3. **Keep.** At each position the model's own token is picked from its prediction, with the normal sampling
   settings; while it equals the guess, the next position is checked too. At the first difference the model's
   token is kept and the rest are undone (`Model::accept`): attention layers just forget the extra keys and
   values; DeltaNet layers, whose running memory can't be rewound, restore it from a per-token snapshot that
   ggml's `gated_delta_net` op keeps for exactly this, along with the convolution's last inputs.
4. **Catch up.** The draft layer re-reads the kept tokens with the model's real hidden states, so its own cache
   matches what the model saw.

Every token that comes out is the model's own choice at that position, so with temperature 0 the reply is the
same as without speculation (the tests check this, also with scripted, partly wrong guesses that exercise every
undo depth), and with sampling it follows the same distribution. The draft layer's math is checked against a
NumPy version of Hugging Face's MTP block.

**KV cache precision.** The cache can be stored at 8 bits (`q8_0`, about 1% from exact) or 4 bits (`q4_0`), which
leaves VRAM for more of the model's layers. Quantized caches are read by flash attention, which turns on with them.

## Using the CPU fully

**One CPU engine per generation.** Release builds ship ggml's engines as separate files: a CPU engine for each x86
generation (plain x86-64, SSE4.2, AVX, AVX2, AVX-VNNI, AVX-512, Zen 4, AMX) plus CUDA and Vulkan. When tierlm starts
(`src/backends.cpp`), ggml scores each CPU engine against the processor and loads the best one; a GPU engine whose
driver isn't installed simply doesn't load. So a 2010 PC runs the plain build and a Ryzen 9000 runs the Zen 4 build
with AVX-512, from the same download. The hardware card shows which instruction set is in use.

**Repacked weights.** Weights the CPU computes with are stored in ggml's repacked layout where one exists (Q4_0,
Q4_K, IQ4_NL, and Q2_K with AVX-512, on x86): several rows interleaved so its matrix kernels read whole blocks at
once. Results are the same (checked to 3e-6); reading a prompt on the CPU was 3.5x faster in testing. Generation is
limited by RAM speed either way. The embedding table, looked up by row rather than multiplied, keeps the plain layout.

## Memory and devices

- **Weights** are read from the GGUF file into a buffer on the chosen device (`ggml_backend_alloc_ctx_tensors`
  plus `ggml_backend_tensor_set`), streamed tensor by tensor.
- **KV cache**: one K and one V tensor per layer, `[head_dim, n_kv_heads, max_ctx]`, stored in **f16** (half
  the size and traffic of f32). Writing new entries is a `ggml_cpy` into a view at the current position;
  attention reads a view of the first `n_past + n_new` positions. Truncating a conversation just moves
  `n_past` back.
- **Scheduler.** Graphs run through `ggml_backend_sched`, with the main device first and the CPU as fallback,
  so any operation a GPU backend can't run is sent to the CPU automatically. Compute memory is reserved once,
  for the largest graph (a full 256-token batch at the end of the context), so it is not re-planned on every
  token. Long prompts are processed in batches of 256 tokens.
- **Routing capture.** When asked, each layer's expert ranking (`argsort` of router probabilities) is marked
  as an output and read back after the step. This is what feeds the dashboard's expert map, and it will feed
  the hot-expert cache.

## Verification

There is no llama.cpp to compare against, so every piece is checked against an independent reference:

| Piece | Reference | Result |
|---|---|---|
| Tokenizer | Hugging Face's OLMoE tokenizer, 14 tricky strings | All exact; all decode back byte-for-byte |
| Pre-tokenizer | The reference split regex, ~3,000 random and edge-case strings | 0 mismatches |
| Model graph | Plain NumPy reference of each architecture on tiny random models (`tests/tiny_models.py`: Llama, Llama 3 with RoPE factors, Mixtral, Qwen2, Qwen3, Qwen3 MoE, OLMoE) | ~2e-6 relative with an f32 cache, including flash attention, grouped-query attention, the KV cache and 300-token batched prompts |
| Hybrid Qwen3.5 | Plain NumPy implementation of Hugging Face's Qwen3_5 forward pass on a tiny random model written the way the converter writes it (`tests/tiny_qwen35.py`), also split between GPU and RAM | ~1e-5 relative with an f32 cache; deliberately wrong file conventions fail |
| Tokenizers without models | Small tokenizers trained with Hugging Face `tokenizers` and `sentencepiece` (`tests/trained_tokenizers.py`) | Exact token IDs for GPT-2, Llama 3 and Qwen2 splitting and SentencePiece |
| Real model | "The capital of France is" | " Paris" at 62–64% on CPU and GPU |

## Design decisions so far

- **Own engine, not a llama.cpp fork or wrapper.** Only ggml's kernels are borrowed.
- **GGUF** as the model format: it covers almost every model on Hugging Face and carries the low-bit
  quantizations tiering depends on.
- **Native Windows Vulkan for Intel Arc**, not WSL's Vulkan-over-DirectX translation. Vulkan also runs on
  AMD and NVIDIA, so it doubles as a universal backend; NVIDIA cards will also get CUDA.
- **Not the NPU.** It is designed for small, fixed, low-power models; it shares the same memory bandwidth (so
  it would not generate faster), handles MoE's per-token routing poorly, and is invisible to WSL.
- **Correctness first**: every change is checked against the references above before it ships.

## What's next

1. **Real-model testing of the new families**, then more families: Gemma, Phi, DeepSeek, gpt-oss. Qwen3-30B-A3B (48 layers, 128 experts, 8 active, about
   18.6 GB in Q4) is larger than the Arc's 16 GB, which makes it the first real test of tiering.
2. **Tier loader, next steps.** VRAM/RAM splitting works (see [TOOLS.md](TOOLS.md#splitting-models-between-vram-and-ram));
   next are the SSD tier (rarely used experts read from disk when routed to) and finer-grained placement.
3. **Hot-expert cache.** Use routing statistics (already captured) to keep the most-used experts in VRAM and
   swap them as usage shifts.
4. **NVIDIA (CUDA) on an RTX 5090** and AMD testing.
5. Quantized KV cache, speculative decoding with draft heads, OpenAI-compatible endpoints, P-core thread
   pinning.
