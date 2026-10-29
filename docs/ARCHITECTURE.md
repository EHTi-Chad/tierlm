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

1. **Guess.** The draft layer runs up to `draft_max` times (default 3 for dense models, 2 for MoE: each guess an MoE model checks brings in its own experts), feeding its own hidden state forward,
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

**Measured.** RTX 3070 laptop, Qwen3.6-27B MTP Q4_K_M with 23 of 64 layers in VRAM, 16K context, 8-bit KV:
3.0 tok/s without speculation; 3.9, **5.3** and 4.1 tok/s with 2, 3 and 4 guesses per check. At 3 guesses a check
produced 2.6 tokens on average and 53% of guesses were kept. More guesses stop paying once the later ones are rarely
kept but still cost a position in the check, so 3 is the default.

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
  the size and traffic of f32), or 8 or 4 bits. New entries are written by index (`ggml_set_rows` with the token
  positions as input), and attention reads the cache in steps of 256 positions, the ones not written yet masked
  out, so the graph keeps the same shapes from one token to the next (see CUDA graphs below). Truncating a
  conversation just moves `n_past` back.
- **Scheduler.** Graphs run through `ggml_backend_sched`, with the main device first and the CPU as fallback,
  so any operation a GPU backend can't run is sent to the CPU automatically. Compute memory is reserved once,
  for the largest graph (a full 256-token batch at the end of the context), so it is not re-planned on every
  token. Long prompts are processed in batches of 256 tokens.
- **Routing capture.** When asked, each layer's expert ranking (`argsort` of router probabilities) is marked
  as an output and read back after the step. This is what feeds the dashboard's expert map, and it will feed
  the hot-expert cache.

## Hot experts (adaptive tiering)

A mixture-of-experts model picks a few experts per token, and some are picked far more than others: on real models a
handful per layer take most of the picks. With experts in RAM, VRAM left after attention, routers and the KV cache
used to hold whole expert layers. Now it holds the **most-used experts of any layer**:

1. **Learning.** Every prompt and reply adds the router's picks to a per-model count (n_layer x n_expert), saved in
   `tierlm-usage/` next to the settings file (counts only). Old habits fade: when the counts get large they're halved.
   **Find the fastest settings** learns from a short mixed text (writing, code, maths, translation) when nothing has
   been learned yet, then measures the result against whole expert layers.
2. **Placing.** At load, experts are ranked by count and added to VRAM until it's full (never-used ones last), so a
   layer can have all, some or none of its experts in VRAM.
3. **Computing a split layer.** Its expert tensors become two: the chosen experts in VRAM and the rest in RAM, each
   with one extra all-zero expert at the end. The router's picks are mapped into each part's own numbering (small
   lookup tables) and get weight 0 in the part that doesn't hold them, so the sum is the same as unsplit. A pick
   that lives in the other part has to point somewhere:
   - **On the CPU** (the RAM part, a few tokens at a time) it gets id -1, which the CPU skips (a small change to
     ggml, see [GGML_CHANGES.md](GGML_CHANGES.md)), so the CPU computes only the experts that are in RAM.
   - **On a GPU** (the VRAM part, and the RAM part for prompts, which ggml hands to the GPU from 32 tokens) GPU
     kernels need valid ids, and CUDA's also need each token's ids to be different. So such picks go to distinct
     experts of that part that the token didn't pick, with weight 0 (a small CPU op, `distinct_ids`). Each part
     therefore holds at least as many experts as a token picks.

   The lookups, the weights and the RAM part run on the CPU and the VRAM part on the GPU, so the work changes device
   twice per layer, as before.
4. **Choosing.** It is off by default: on an RTX 3070 laptop with Qwen3.6-35B-A3B the VRAM experts did 71% of the
   expert work, but the plain split was still faster (33.3 against 29.7 tok/s), because the split adds copies and
   bookkeeping per layer. **Find the fastest settings** measures both and turns it on only where it wins.
   With the timing switch on (`TIERLM_PROFILE=1`) the server prints where each token's time goes.

The tests run every MoE model with made-up usage counts at many VRAM sizes, so layers end up whole, split or absent,
and compare with the reference (exact on the stand-in GPU).

## Several GPUs

With `device` `auto` or `gpu`, every GPU of the same kind as the best one (the same engine: CUDA, Vulkan...) becomes
another VRAM tier, so one card that shows up under both CUDA and Vulkan isn't counted twice. Whole layers fill the
first GPU from the last layer backwards, then the next GPU, then RAM, each layer's ops pinned to its device. With
experts in RAM, each layer's attention, router and KV cache go on a GPU first (again filling one before the next),
then the experts: on their layer's GPU if it has room, else another GPU, else RAM. The tests run every model split
over two stand-in GPUs (ggml's RPC backend) and RAM at many memory sizes, and compare with the reference.

## CUDA graphs

ggml can record the GPU kernels of a graph once and replay them with a single launch, but its build option for this
(`GGML_CUDA_GRAPHS`) is off unless turned on, and a recording is only replayed while the graph's shapes and data
pointers stay the same. tierlm turns it on and keeps its graphs steady: the KV cache is written by index and read in
steps of 256 positions (above), and one-token steps, several-token steps and the MTP draft head are built in separate
memory so each keeps its own recording. Without it, a laptop RTX 3070 spent about 22 ms per token mostly launching
small jobs; with it, Qwen3.6-35B-A3B with experts in RAM went from 21.7 to 33.3 tok/s, with the same results.
Details: [GGML_CHANGES.md](GGML_CHANGES.md).

## Auto-tune

`/api/autotune` measures what only changes speed, one setting at a time in the order that matters most: the split
mode (a reload each), then CPU threads, then speculative decoding with 1 to 4 guesses. Since 0.5.1 each measurement
runs exactly what a chat runs: the chat's own reply code, in the model's chat format, with the chat's sampling
settings (temperature, top-k, top-p, min-p, repeat penalty), recording the experts used, answering two different
questions for about 3 seconds each. A fixed seed makes every setting answer the same way, so they are compared
fairly. Only the replies are timed. Threads and speculation only win by a clear margin (2-3%), so noise doesn't
pick them, and hot experts are kept on only if they measured fastest. The **Benchmark** in the side menu measures
the same way. (Before 0.5.1 it generated a fixed text greedily, which overstated chat speed by a quarter to a third;
part of that gap was the chat reading each layer's experts back from the GPU separately, now one read per token.)

## CPU and GPU at the same time

With plain experts in RAM there is little to overlap: in each layer the CPU's experts need the GPU's attention and
router, and the GPU's next layer needs the experts' result. With hot experts, a split layer's VRAM experts (and the
shared expert) could run on the GPU while the CPU computes the RAM experts. ggml's scheduler runs device splits one
after another and waits at every hand-off, so this needs changes there.

## Verification

There is no llama.cpp to compare against, so every piece is checked against an independent reference:

| Piece | Reference | Result |
|---|---|---|
| Tokenizer | Hugging Face's OLMoE tokenizer, 14 tricky strings | All exact; all decode back byte-for-byte |
| Pre-tokenizer | The reference split regex, ~3,000 random and edge-case strings | 0 mismatches |
| Model graph | Plain NumPy reference of each architecture on tiny random models (`tests/tiny_models.py`: Llama, Llama 3 with RoPE factors, Mixtral, Qwen2, Qwen3, Qwen3 MoE, OLMoE) | ~2e-6 relative with an f32 cache, including flash attention, grouped-query attention, the KV cache and 300-token batched prompts |
| Hybrid Qwen3.5 | Plain NumPy implementation of Hugging Face's Qwen3_5 forward pass on a tiny random model written the way the converter writes it (`tests/tiny_qwen35.py`), also split between GPU and RAM | ~1e-5 relative with an f32 cache; deliberately wrong file conventions fail |
| Tokenizers without models | Small tokenizers trained with Hugging Face `tokenizers` and `sentencepiece` (`tests/trained_tokenizers.py`) | Exact token IDs for GPT-2, Llama 3 and Qwen2 splitting and SentencePiece |
| Splits | Every model split between one or two stand-in GPUs (ggml's RPC backend) and RAM at many VRAM sizes, by layers, by experts and with made-up hot-expert usage | Same results as unsplit (~1e-6 with an f32 cache) |
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

1. **CPU and GPU on the same layer at once**: with hot experts, a split layer's VRAM experts and RAM experts could run
   in parallel; ggml's scheduler runs them one after the other today. Also moving experts between VRAM and RAM while
   the model is loaded, as usage shifts, instead of at the next load.
2. **SSD tier**: rarely used experts read from an NVMe drive when the router picks them, for models bigger than VRAM
   and RAM together.
3. **Usage-based precision** (rarely used experts stored at lower precision) and **expert-aware speculation** (guess
   further ahead only while the guesses reuse experts that are already loaded).
4. **More families and real-model tests**: Gemma 4, Phi, DeepSeek, gpt-oss; Qwen3-Coder-Next, Mixtral and Qwen2.5 on
   real weights. Tool calling in the API.
5. Prompt-lookup guessing for models without a draft layer, and the Zen 4 / AMX CPU engines on Windows.
