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

1. **Real-model testing of the new families**, then more families: the Qwen3.5 hybrid (Gated DeltaNet
   linear attention), Gemma, Phi, DeepSeek, gpt-oss. Qwen3-30B-A3B (48 layers, 128 experts, 8 active, about
   18.6 GB in Q4) is larger than the Arc's 16 GB, which makes it the first real test of tiering.
2. **Tier loader, next steps.** VRAM/RAM splitting works (see [TOOLS.md](TOOLS.md#splitting-models-between-vram-and-ram));
   next are the SSD tier (rarely used experts read from disk when routed to) and finer-grained placement.
3. **Hot-expert cache.** Use routing statistics (already captured) to keep the most-used experts in VRAM and
   swap them as usage shifts.
4. **NVIDIA (CUDA) on an RTX 5090** and AMD testing.
5. Quantized KV cache, speculative decoding with draft heads, OpenAI-compatible endpoints, P-core thread
   pinning.
