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
│   ├── encoder       embedding models (BERT, Nomic)
│   ├── vision        the vision encoder (Qwen3.5 / 3.6 mmproj files); image: decoding images
│   └── model         one compute graph for every supported architecture, KV cache, flash attention,
│                     batching, device placement
├── tools/         command-line tools and the web server
├── web/           the web page and dashboard (plain HTML/JS, no build step)
├── tests/         reference tests (Python, testing only)
└── third_party/
    ├── ggml/      math kernels, quantized formats, device backends, GGUF reader  (MIT, submodule)
    ├── httplib/   HTTP server                                                    (MIT, single header)
    ├── json/      JSON                                                           (MIT, single header)
    └── stb/       stb_image: decoding PNG / JPEG / GIF / BMP          (public domain or MIT, single header)
```

tierlm deliberately does **not** use llama.cpp. ggml supplies the tensor operations, the same way a game
engine relies on DirectX or Vulkan: fast matrix multiplies on quantized weights, rotary embeddings, flash
attention kernels, and backends for CPU, Vulkan and CUDA. Everything that decides *what* to compute and
*where* things live is tierlm's.

## How one token is produced

1. **Tokenizer.** Text is split into words, numbers, punctuation and whitespace (a hand-written version of
   the model's split pattern: GPT-2, Llama 3, Qwen2/3.5 or o200k, using generated Unicode tables), each piece's bytes are mapped to GPT-2's byte
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

## gpt-oss

OpenAI's gpt-oss models (and models built on them, such as HyperNova-60B) differ from the other families in a few
places, each checked against a NumPy reference written from Hugging Face's `modeling_gpt_oss.py`:

- **Attention sinks**: each head has one learned extra score that takes part in the softmax but points at nothing,
  so a head can attend "nowhere". ggml supports them in flash attention and in its softmax (`*_add_sinks`).
- **Sliding window**: every other layer (the even ones) sees only the last 128 tokens; those layers get their own
  mask. Their KV cache is still full length for now.
- **Biases** on Q, K, V and the output projection, on the router, and on every expert (`ggml_add_id` adds each
  pick's own expert's bias).
- **Routing**: the top 4 experts by score, then a softmax over just those four.
- **Experts**: a clamped SwiGLU, `(clamp(up, -7, 7) + 1) * g * sigmoid(1.702 g)` with `g = min(gate, 7)`
  (`ggml_swiglu_oai`), on MXFP4 weights.
- **YaRN, exactly**: gpt-oss stretches its 4,096-position training to 131,072 with YaRN. ggml's built-in YaRN rounds
  the band of dimensions that are blended to whole dimensions; gpt-oss is configured without that rounding
  (`"truncate": false`), and so is Hugging Face's reference. tierlm computes the blended frequencies itself, as
  Hugging Face does, and gives them to ggml's rope as per-dimension frequency factors (the mechanism Llama 3.1 uses),
  with the cos/sin scale `0.1 ln(factor) + 1`. The test model's band edges fall between whole dimensions, so the
  rounded version would fail it.
- **Tokenizer**: OpenAI's o200k split pattern, which also splits words where lower case turns to upper case
  (hand-written like the others; the tests compare it with the pattern itself on 3,000 strings).
- **Chat format ("harmony")**: a reply is several messages, the reasoning on the `analysis` channel and the answer on
  the `final` channel, or a tool call on `commentary` addressed `to=functions.NAME`. tierlm builds the prompt exactly
  as OpenAI's chat template does (tools as a TypeScript namespace; `tests/harmony_test.py` compares them byte for
  byte), stops only at `<|return|>` or `<|call|>`, and shows the reasoning as the usual collapsible "Thinking".

Not yet for gpt-oss: hot experts (the split doesn't divide the expert biases yet; whole expert layers go to VRAM
or RAM instead) and a shorter KV cache for the sliding-window layers.

## Several conversations in the cache

A chat's whole conversation is sent with every request, and tierlm only reads the part that differs from what is
already in the KV cache. With one cache, any request for a different conversation throws the first one away: a
coding agent's side request (opencode titling a session, with its own short instructions) used to cost the next real
request a full re-read of the agent's 10,000+ tokens of instructions, about two minutes on a laptop.

So when a request would overwrite at least 256 tokens of the cached conversation, tierlm first copies that
conversation to system RAM (`Model::save`: the keys and values of its positions in every attention layer, which are
one contiguous block per layer, plus the DeltaNet layers' running memories and the MTP draft block's state). When a
later request continues a saved conversation further than the cache does, the saved one is copied back
(`Model::restore`) and only the new tokens are read. Up to 8 conversations are kept, within 4 GB and never more
than a quarter of the RAM that is free, least recently used dropped first. Hybrid models (Qwen3.5 / 3.6) can't
rewind their running memory, so for them a saved conversation is used only when the request continues all of it.

Measured on Qwen3-0.6B on 2 CPU threads: a 2,500-token agent prompt, a side request, then the next turn: 4.5 s
instead of about 45 s, the saved conversation copied back in 0.05 s, and the reply word for word the same as without
the side request. The tests save every model family half way through a conversation, read a different one, restore
the first and finish it, and compare with reading it straight through (identical, hybrid models included).

**Follow-up messages on hybrid models.** A hybrid model can't rewind, so it could only reuse the cache if the new
request repeated it token for token, and the last few tokens never did: the cache ends with the reply as the model
wrote it, after the prompt's own `<|im_start|>assistant` header (with Thinking off, an empty `<think></think>` block),
while the history came back as text without that block, and text can split into tokens differently from how the
model wrote it. So every follow-up read the whole conversation again (2,234 tokens, about 20 s, on an RTX 5090 with
Qwen3.8 Flash-Next). Three changes fix it:

- **A point to go back to.** The prompt is read in two parts: everything before the reply's own turn starts, then
  the turn's header. In between, `Model::checkpoint` copies the DeltaNet running memories (and the PLE and MTP
  states) at that point, and `Model::truncate` can go back to it instead of starting over.
- **History written as the model saw it.** With Thinking off, earlier replies are written back with the empty
  thinking block they were generated after.
- **The model's own tokens.** When the conversation's text continues all of the cache's text, the cache's tokens
  are kept (the same text) and only what follows is split into tokens.

Now a follow-up reads only the new message: 4,391 of 4,415 tokens reused on the RTX 5090, and on the laptop with
Qwen3.6-35B-A3B 176 of 208 tokens (the reply and everything before it). `TIERLM_TRACE_CACHE=1` prints where a
request stops matching the cache.
The idea of keeping several conversations' prompt caches comes from other engines: llama.cpp's server keeps several
slots, and Magnitude shares cached prompts between concurrent agent sessions.

## Embedding models (a second model)

Search and document-comparison tools need an embedding model next to the chat model. These are encoders (BERT-style:
`bert`, `nomic-bert`), small enough to live whole on one device, so they get their own small engine
(`src/encoder.cpp`) instead of the tier loader:

- **The graph**: token embeddings (plus learned position embeddings for BERT; Nomic rotates Q and K instead, NeoX
  layout), a LayerNorm, then per layer attention in **both directions** (no causal mask, no KV cache: the whole text
  is read in one pass), a residual and LayerNorm, the feed-forward block (BERT: exact erf GELU with biases; Nomic:
  SwiGLU without biases), a residual and LayerNorm. The last layer's states are pooled as the file says: averaged
  (`mean`), or the first token's (`cls`).
- **The tokenizer**: WordPiece, written for tierlm like the others. Text is cleaned (control characters dropped,
  spaces normalised, CJK ideographs spaced out), uncased models fold each character to lower case without accents
  (Unicode NFD minus combining marks; tables generated from the Unicode database, Hangul computed), punctuation is
  split off, and each word is cut into the longest vocabulary pieces from the left.
- **Placement**: on the GPU when the model fits with plenty to spare after the chat model, else on the CPU.
- **Sharing the GPU**: the chat model and the embedding model take one lock per step (`device_mutex`), so an
  embedding request waits for at most one token of a chat reply, not the whole reply. Measured on the CPU with a
  13-second reply streaming: each embedding request answered in about 0.06 s.

Checked like everything else: `tests/tiny_encoders.py` compares tiny random `bert` (CLS and mean pooling) and
`nomic-bert` models with a NumPy forward pass (worst difference 4e-7), and the WordPiece tokenizer with Hugging Face's
own BERT tokenizer, uncased and cased, on strings with accents, CJK, Korean, Turkish, control characters and words too
long for the vocabulary (all identical).

## Images (the vision encoder)

Qwen3.5 / 3.6 read images with a vision encoder stored in a separate `mmproj-*.gguf` next to the model. tierlm loads
it with the model (`src/vision.cpp`: on the GPU when there's room after the model; else its weights stay in RAM and
a GPU with a little room still does the heavy work, ggml copying the weights over op by op ("op offload", worth it
because an image is thousands of patches); else the CPU) and follows Qwen's
own code (Hugging Face transformers' `Qwen3_5VisionModel` and `Qwen2VLImageProcessor`):

- **Resize**: the image keeps its shape; both sides become multiples of 32 and the pixel count stays within the
  limit (2,048 "image words" by default: one per 32 x 32 pixels), with an antialiased bicubic filter. Values are
  scaled to -1 .. 1.
- **Patches**: 16 x 16 squares, listed 2 x 2 block by block (the order the merge step needs). The patch layer is a
  3D convolution over two copies of the image (Qwen treats an image as a 2-frame video); with both frames the same
  that is one linear layer with the two halves of the kernel added, done once at load.
- **Positions**: a learned 48 x 48 table, bilinearly resampled to the image's grid (corners aligned), is added; in
  the blocks, 2D rotary positions (row and column, a quarter of each head each, ggml's vision rope).
- **Blocks**: LayerNorm, attention over the whole image (flash attention with 16-bit keys and values for big
  images, more than 1,024 patches, so a scanned page doesn't need N x N scores per head), GELU (tanh form, written
  out: ggml's own reads a 16-bit table on the CPU) MLP.
- **Merge**: LayerNorm, each 2 x 2 block becomes one vector, and an MLP maps it to the chat model's width.

**In the chat model** (`Model::add_image`): an image's words are registered once (keyed by a fingerprint of the
file, so the same image sent again with each turn isn't read again) and fed like tokens, as ids above 2^30 that
the step swaps for the words. Qwen gives positions in 3 parts (image, row, column), interleaved over the rotated
dimensions (`rope.dimension_sections`, ggml's IMROPE): text has all three equal, which is plain RoPE, so text-only
steps are unchanged; an image's words get (start, start + row, start + column), and text after it continues at start
+ max(rows, columns), as Qwen's `get_rope_index` does. The attention mask still follows the order of the tokens.
Each conversation keeps its tokens' positions, so cached and saved conversations, rewinding and several
conversations at once work with images too; an image cut across prompt chunks finds its start in the earlier part.

Checked by `tests/tiny_vision.py` against Hugging Face's own Qwen3_5ForConditionalGeneration (PyTorch) on a tiny
random model: patches identical; image words within 7e-7 (3e-5 with flash attention on a 1,152-patch image); a
model reading images between text within 1e-4 of Hugging Face's logits (the text-only difference of the two
implementations), whether the prompt is fed at once, a token at a time or in pieces that cut an image in two; the
resize within about 1 of 255 levels of PIL's.

## Flint

Recursive's Flint (40B-A2.5B, a coding model) mixes pieces tierlm already had: attention sinks and sliding-window
layers (as gpt-oss, a 513-token window), per-head QK-norm (as Qwen3), and a shared expert (as Qwen3.5 MoE, but added
in full, without a gate). What's new: each expert is scored by its own **sigmoid** (not a softmax over all of them),
the top 6 of 256 are picked and their scores renormalised; the **first layers are dense** (an ordinary feed-forward
block, no experts); and only the sliding-window layers **rotate positions**: the full-attention layers use none.
Checked against a NumPy version of Hugging Face's `modeling_flint.py` (a dense layer, a sliding MoE layer and a full
MoE layer; 2e-6), also split between a stand-in GPU and RAM. Its tokenizer's word splitting and its chat format
(GLM-style, `<|user|>`) still need the real model files to check against.

## Speculative decoding

When a model doesn't fit in VRAM, generating a token is limited by how fast its weights in RAM can be read: a dense
27B model with 14 GB in RAM manages about 3 tokens/second on a laptop. Speculative decoding gets several tokens
out of each read. Qwen3.5 / 3.6 / 3.8 and Qwen3-Next ship a small **MTP draft layer** (multi-token prediction):
one full-attention block that takes the token just chosen and the model's last hidden state, and predicts the
next token. Unsloth publishes it as a separate `mtp-<model>.gguf`; tierlm loads it from either place.
Qwen3.8 Flash-Next's draft block works on its hyper-connection copies: it reads the model's copies before the last
stream mixer (`hidden_width()` = copies x width), puts each one beside the token's embedding through `eh_proj`, runs
one full-attention MoE block with its stream mixers, and mixes the copies back with its own last mixer
(`nextn.hc_head_*`). Unsloth's "shared" file leaves out the token table and uses the model's.

Each step (`src/speculate.cpp`):

1. **Guess.** The draft layer runs up to `draft_max` times (default 3 for dense models, 2 for MoE: each guess an MoE model checks brings in its own experts), feeding its own hidden state forward,
   which is cheap: one layer, usually in VRAM.
2. **Check.** The model evaluates the chosen token and the guesses in **one pass** (`Model::eval_ex`), producing a
   prediction for every position. Its weights are read once for all of them.
3. **Keep.** At each position the model's own token is picked from its prediction, with the normal sampling
   settings; while it equals the guess, the next position is checked too. At the first difference the model's
   token is kept and the rest are undone (`Model::accept`): attention layers just forget the extra keys and
   values; DeltaNet layers, whose running memory can't be rewound, restore it from a per-token snapshot that
   ggml's `gated_delta_net` op keeps for exactly this, along with the convolution's last inputs. Flash-Next's
   per-layer embedding keeps a short convolution of its own, rewound the same way (since 0.8.5; until then
   Flash-Next didn't guess at all).
4. **Catch up.** The draft layer re-reads the kept tokens with the model's real hidden states, so its own cache
   matches what the model saw.

Every token that comes out is the model's own choice at that position, so with temperature 0 the reply is the
same as without speculation (the tests check this, also with scripted, partly wrong guesses that exercise every
undo depth). The draft layer's math is checked against a NumPy version of Hugging Face's MTP block (Flash-Next's,
against one built from Hugging Face's own layers).

**Sampling** (temperature > 0, since 0.8.5). Checking the model's own draw against the draft layer's single best
guess keeps few guesses when the model is really choosing between words: a story at temperature 0.7 kept under half.
So with sampling, tierlm uses **speculative sampling**: each guess is drawn from the draft layer's own probabilities
q (after the same penalties, temperature and top-k/top-p/min-p as the reply), and kept with probability
min(1, p/q), where p is the model's. At the first one not kept, the token is drawn from what p has left over q
(p - q where positive, renormalised); when every guess is kept, one more token from p. The reply's odds are exactly
the model's own: `tests/tiny_flashnext_mtp.py` draws 8,000 three-token replies with and without it on a
16-token vocabulary (40% of guesses kept) and compares them with a chi-square test, token by token, in pairs and
whole. Flash-Next IQ3_XXS on an RTX 5090, 4 guesses: a story at 0.7 went from 46-52 to 63 tok/s (55 without
guessing); prompts at 0.7 from 67-73 to 77-90.

**Kept plans** (since 0.8.7). A round of guessing runs three kinds of graph in turn: the draft block for one
token (once per guess), the model's check, and the draft block catching up. ggml's scheduler holds one placed graph
at a time, so each of them was built and placed again every round, and the reply's one-token graph that tierlm
keeps to run again (about 3 ms a token saved) never survived a round. Measured on Flash-Next with `TIERLM_PROFILE`:
1.0 ms to build and 2.7 ms to place the check's graph, of a 45 ms check, and the same kind of work again for each
of the draft block's steps. Now the check and the draft block's one-token step each have a scheduler of their own,
with its own working memory, where their graph stays placed and is run again with new inputs (positions, cache
cells and masks are inputs already), until the cache span or the number of guesses changes. Undoing guesses reads
the snapshots straight from the check's own memory, which nothing else writes over.

The cost is VRAM: a check keeps a copy of every DeltaNet layer's memory per guess, so its working memory is large
(571 MB for Flash-Next with 4 guesses, where the largest prompt batch needs 548 MB). It is counted with the VRAM
kept free at load, so that many fewer experts fit: 10,729 instead of 11,005 of Flash-Next's 24,576 on an RTX 5090,
still 90% of the expert work. The plans are made only for models with a draft block, and only when the check's
memory is under 1/32 of the card (`TIERLM_PLANS=1`: always, `=0`: never).

The draft block's experts also go into VRAM before the model's own (since 0.8.7; `TIERLM_MTP_VRAM=0` for the old
placement by use): the draft block runs once per guess, so its experts in RAM cost a GPU-to-CPU round trip each time.

RTX 5090 (32 GB), Ryzen 9 9950X3D, 64 GB DDR5-6000, Windows 11, Flash-Next UD-IQ3_XXS with its MTP file, 64K
context, 12 threads, VRAM experts following the conversation, 4 guesses, 500-token replies, one session:

| | Four answers, temperature 0 | A story at 0.7 | Guessing, per check | Checking, per check |
|---|---|---|---|---|
| 0.8.6 placement of the draft block's experts | 65.0 tok/s average | | 7.4 ms | |
| Draft block's experts in VRAM first | 71.7 | | 3.9 ms | |
| (later session) without kept plans | 77.3 (69.3-87.4, 16 replies) | 57.5 (54.5-61.3) | 3.5-4.5 ms | 33.5-41.6 ms |
| With kept plans | **86.5** (76.1-96.6, 24 replies) | **66.5** (62.9-69.9) | 2.7-2.9 ms | 29.3-39.1 ms |

The two pairs of rows are from different sessions (a session's speeds depend on how much of the n-gram table Windows
has cached), so compare within a pair. Replies are identical with and without kept plans (`tests/tiny_flashnext_mtp.py`,
`tests/tiny_qwen35.py`).

Also measured and not kept: choosing the number of guesses per check from how sure the draft block is (stop when
the chance that a guess is kept, calibrated per place as the reply goes, falls under what one more checked position
costs). With at most 4 guesses it matched a fixed 4 on answers (76.9 against 77.3 tok/s) and was slower on a story
(54.5 against 57.2); allowed up to 8 it was slower everywhere (65.3 on answers), because the model's check gets
slower per position past 4. A fixed count stays.

**Measured.** RTX 3070 laptop, Qwen3.6-27B MTP Q4_K_M with 23 of 64 layers in VRAM, 16K context, 8-bit KV:
3.0 tok/s without speculation; 3.9, **5.3** and 4.1 tok/s with 2, 3 and 4 guesses per check. At 3 guesses a check
produced 2.6 tokens on average and 53% of guesses were kept. More guesses stop paying once the later ones are rarely
kept but still cost a position in the check, so 3 is the default.

**Prompt lookup.** A second source of guesses, for every model, with or without a draft layer: when the last 3 to
6 tokens also appear earlier in the conversation, the tokens that followed them there are the guess
(`Speculator::lookup_guesses`, searched from the most recent place back). Replies often repeat what's already in the
conversation: a file being rewritten with a small change, code being quoted, names and paths. Those guesses go
through the same check and undo as the draft layer's, so the reply is the same. Lookup guesses are used when there is
a match, the draft layer's otherwise. The guess length adapts: after a guess the model rejected at once, the next
lookup guess is a single token (cheap to check); after a fully kept one, twice as long, up to the maximum, which
depends on where the model lives: 8 with everything in VRAM, 6 for hybrid models (each guessed token needs an undo
copy of the DeltaNet memory, about 60 MB for Qwen3.6-35B-A3B), 4 for MoE models with experts in RAM (each guessed
token picks its own experts, which the CPU has to read). The tests check long guesses (8 at a time, wrong at
different depths) on every hybrid variant.

Lookup also checks that it pays for itself. A checked guess is not free: on a MoE model each guessed token reads its
own experts, so a wide check can take much longer than one token, and a miss gets nothing for it. Reasoning text
matches earlier text often (the question's numbers, the model's own markers) but continues differently, so on a fast
GPU those misses added up: gpt-oss-20b on an RTX 5090 wrote a short worked answer at 135 tok/s with lookup and 234
without. So the speculator times every step: plain steps (no guess) keep a running average, and lookup checks are
added up in rounds of four. When a round gave fewer tokens per second than plain steps would have, lookup pauses (16
steps, then twice as long each time it loses again, back to 16 when it wins) and tries again later, in case the reply
starts repeating text. After the change: 270 tok/s on the same question, while rewrites keep their gain (Qwen3-0.6B
on 2 CPU threads, a rename in 1.6 KB of code: 12.5 → 18.6 tok/s; a short maths answer: 15.5 → 15.0, within noise).
The timing decides only whether to guess, never what the reply says.

**Measured** (Qwen3-0.6B Q8_0 on 8 CPU threads, temperature 0, same reply with and without): rewriting 2.6 KB of
code with one word renamed, 10.5 → **27.7 tok/s** (6.6 tokens per check, 203 of 212 guesses kept); writing new
code, 12.1 → 12.1 tok/s (lookup found few matches and the adaptive length kept misses cheap).

**Flash-Next on the RTX 5090** (UD-IQ3_XXS, 25.7 GB of experts in RAM, busiest experts in VRAM, Ryzen 9 9950X3D,
64 GB DDR5-6000, 12 threads, Unsloth's `mtp-...-shared-Q8_0.gguf`, 4 guesses, four 500-token answers at
temperature 0): 61.6 tok/s without guessing, **84.3** with it, 3.1-3.6 tokens per check. A check of 4 guessed words
costs about 35 ms against 18 ms for one word: each distinct expert the words pick is read once, and most of them are
already in VRAM. With the experts in RAM instead, checks cost too much and guessing lost (47 against 48 tok/s), which
is why Find the fastest settings measures the VRAM-expert splits with guesses too.

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
- **VRAM kept free.** On Windows a GPU that runs out of memory quietly borrows system RAM through the driver, and
  the work on it gets about 20 times slower (Qwen3.8 Flash-Next fell to 1.2-1.7 tok/s on an RTX 5090). So the
  loader keeps VRAM free for the GPU's working memory: after loading, it reads how big the scheduler's compute
  buffers really are, adds scratch for the GPU engine and the driver (1/32 of the card, 192 to 640 MB:
  `TIERLM_VRAM_SCRATCH_MB`), and if that is more than it kept, loads once more with that much kept, remembering it
  per model, context, batch and split kind next to the learned expert use (`<usage>.vram`). Up to 0.8.3 the scratch
  was 640 MB on every card; on an RTX 3070 Laptop (8 GB) with Qwen3.6-35B-A3B, 400 MB kept in all (296 of it the
  compute buffers) never used shared memory over a 1,500-token reply (the process's shared GPU memory stayed at
  98 MB, the card still had 1 GB free) and fit 2,865 VRAM experts instead of 2,459 with 936 MB kept: 40.0 against
  38.7 tok/s.

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

### Live hot experts (following the conversation)

The learned set describes the model's average use, but a conversation has its own: a coding session leans on
different experts than an email. With **Let the VRAM experts follow the conversation** (`live_experts` in
`/api/load`, also tried by **Find the fastest settings** as a fourth split mode), the VRAM experts keep changing
while the model runs:

- **Experts trade places.** An expert moving into VRAM takes the slot of the one it replaces, and that one goes
  into the RAM slot the newcomer left: three copies per expert tensor (the RAM slot to a buffer, the VRAM slot to the
  RAM slot, the buffer to the VRAM slot). The RAM part stays exactly the size of the fixed set's, so the mode needs
  no extra RAM and works on any PC that can load hot experts. (Up to 0.8.2 the RAM part held every expert, the VRAM
  ones twice, and the mode was turned off when RAM couldn't spare that: on a 64 GB PC with Qwen3.8 Flash-Next it
  never ran.) Each split layer keeps two maps, expert to VRAM slot and expert to RAM slot, and the lookup tables the
  graph reads change with them; an expert not in a part points at that part's zero expert.
- **Kept in the file's layout.** The RAM part stays in the model file's layout (not ggml's repacked one, which can't
  be read back), so each expert is one slice to copy.
- **Scores.** Each split layer keeps one score per expert. It starts from the learned counts (scaled so the busiest
  counts as about 3 recent picks; with nothing learned yet, every expert starts even, so live mode works from the
  first load). After every step each pick adds 1, and all scores fade by 3% per token, so the last few dozen tokens
  matter most.
- **Swapping.** After each step, the best-scoring expert in RAM replaces the worst-scoring one in VRAM if it
  clearly beats it (more than 1.3x its score + 0.5). Without that margin, two experts of about equal use would
  keep trading places. At most 2 swaps per layer per step (`TIERLM_LIVE_PER_LAYER`); each step starts at a different
  layer, so every layer gets its turn. The copy happens after the step's work has finished and under the device
  lock, so no running graph is reading the slot. The copies are queued on the GPU with one wait at the end of the
  step, not one per copy.
- **How much to move.** Moving costs time on every word, so each step may spend about 8% of a step's time on it
  (`TIERLM_LIVE_SHARE`), worked out from how long recent steps took and how fast recent moves went, at most 32 MB
  (`TIERLM_LIVE_MB`); a prompt, being a long step, may move up to that limit. Up to 0.8.3 the limit was a flat 32 MB
  per word, which on an RTX 3070 laptop spent a quarter of the time moving experts: on five subjects with
  Qwen3.6-35B-A3B (300 tokens each, no speculative decoding) that ran at 28.1 tok/s, slower than the fixed set's
  30.7. Measured budgets, in later runs on the same laptop: 4 MB per word 40.2 tok/s, 8 MB 40.5, 16 MB 38.4,
  against 35.7-38.4 for the fixed set; then the share of the step's time: 5% 40.5, 8% 40.7, 12% 40.6, fixed set
  38.0-38.4. On the RTX 5090 with Qwen3.8 Flash-Next (four subjects, 500 tokens each, two runs each): 8% of the
  time 38.8 and 38.8 tok/s with 0.9-1.1 s of moving per reply, 0.8.3's 32 MB per word 35.6 and 39.2 with 1.3-1.8 s,
  the fixed set 31.5 and 33.8.
- **Same answers.** Each pick is still computed in whichever part holds it, so the result equals the unsplit
  model, whichever experts are in VRAM at the time. The tests run every MoE model (and Qwen3.5/Qwen3-Next with
  speculation and saved conversations) with `TIERLM_LIVE_EAGER=1`, which swaps at every chance, and compare with the
  reference.
- **Watching it.** `/api/info` reports `live`: picks, picks served from VRAM, swaps and time spent copying. Model
  info shows the share served from VRAM, and the expert map's rings move as experts come and go.

Measured with four questions on different subjects (C++ templates, black holes, Ontario contract law, sourdough;
500 tokens each, temperature 0), against the fixed set learned mostly from stories:

| PC | Model | Fixed set | Following the conversation |
|---|---|---|---|
| RTX 5090 (32 GB), Ryzen 9 9950X3D, 64 GB DDR5-6000 | Qwen3.8 Flash-Next Q4_K_M (120 GB: attention and 7,260 of 24,576 experts in VRAM, the rest of the experts in RAM, the n-gram table on an NVMe drive) | 35.9 tok/s (31-43 by subject); VRAM experts did 35-69% of the expert work | 40.2 and 39.7 tok/s in two runs (38-41 by subject); 73-75% |
| RTX 3070 Laptop (8 GB), 32 GB DDR4-3200 | Qwen3.6-35B-A3B UD-Q3_K_XL (attention and the busiest experts in VRAM, the rest in RAM), 6 threads, 300 tokens each, five subjects | 25.7 tok/s; 28-48% | 32.2 tok/s; 58-76% |

Moving the experts cost about 1.2 s per 500-token reply on the RTX 5090 and 1.3-2.9 s per 300-token reply on the
laptop (0.8.3, before the budget became a share of each step's time, about 0.5 s on the laptop since), which is why
the fixed set can still win by a little on a subject it already suits (43.1 against 41.0 on black holes on the
5090). **Find the fastest settings** measures both, without letting its own test questions teach the fixed set,
with replies of at least 160 tokens, and with speculative decoding on the runner-up when it was within 15%.

This is our own design, inspired by the expert VRAM cache that landed in llama.cpp this autumn (`--moe-cache-mib`,
off by default there). The difference: theirs starts empty and fills from what's picked. Ours starts from what
tierlm learned about the model, adds a fading score with a swap margin, and trades places with the expert leaving
so nothing is held twice.

## Several conversations at once

With `LoadOptions::parallel` > 1 the model holds several conversations and `eval_seqs` advances any of them in one
step, so the weights are read once for all of them (where the gain comes from: one token's step is mostly reading
weights).

- **One cache, shared by position.** The KV cache has `max_ctx` places ("cells"). Each conversation takes cells as
  it grows, lowest free first, and gives them back when it ends or is cut short; a table records each cell's
  conversation and position. So a short chat holds little memory, and what one finishes the others can use at
  once (the benefit of vLLM's paged cache, without fixed pages). One conversation alone gets cells 0, 1, 2...: its
  positions, as before.
- **The mask keeps them apart.** Keys and values are written to their cells by index; each token's attention mask
  shows only its own conversation's cells up to its own position (sliding-window layers: only the last `n_swa` of
  those). Positions for RoPE are each conversation's own.
- **DeltaNet layers** (Qwen3.5, Qwen3-Next) keep one running memory per conversation. A batch is laid out
  conversation by conversation; the projections run once for all tokens, then the convolution and the memory
  update run per conversation with its own memory, and the outputs are joined again.
- **Each conversation's last token** gives its logits (picked out with one lookup).
- Conversation 0 is also what `eval`, `eval_ex`, speculative decoding, the MTP block and saved conversations use;
  saving and restoring copy its cells run by run, wherever they are.

Tested like everything else: `tierlm-logits --parallel 3` runs three conversations made from the same ids (one
through the one-conversation path, the others together, with a throwaway conversation leaving a hole in the cache,
then mixed steps of 1 and 3 tokens), and each must match the reference on its own, for every model family, split
between the stand-in GPU and RAM, with hot experts and live swapping. `tierlm-bench --parallel 1,2,4,8 ctx`
measures the total and per-conversation speed.

**In the server** (the **Connections** setting, `"connections"` in `/api/load`): the cache holds `ctx x connections`
cells and each connection is a slot (a conversation). Each reply still runs on its own thread with the same code as a
single reply (chat formats, tools, stop strings, sampling with its own random generator); only its model steps go
through a **batcher**: a thread that waits a moment (up to 3 ms) for every running reply to ask for its next step,
then computes them all in one `eval_seqs` and hands each its logits. A long prompt is fed 256 tokens per step, so a
new request doesn't stall the others' words. A reply takes the slot whose cache already holds the longest start of
its conversation. Replies take the engine lock shared (a model load takes it whole, so it waits for them), and one
free connection; the fair queue lets in as many as there are connections. Speculative decoding and the saved
conversations stay with single replies. **Find the fastest settings** measures 1, 2, 4… replies at once and
recommends the most that keeps the slowest reply at or above a floor (5 tok/s by default). Same answers: with greedy
sampling, four replies written at once came out identical to the same four written one at a time.

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
pick them, and hot experts are kept on only if they measured fastest. With a draft layer, the VRAM-expert splits
(busiest, or following the conversation) within 15% of the winner without guessing are measured with guesses too:
guessing makes checks cheap only when most experts are in VRAM, so a split that lost without guessing can win with
it (0.8.5). The **Benchmark** in the side menu measures
the same way. (Before 0.5.1 it generated a fixed text greedily, which overstated chat speed by a quarter to a third;
part of that gap was the chat reading each layer's experts back from the GPU separately, now one read per token.)

## CPU and GPU at the same time

With plain experts in RAM there is little to overlap: in each layer the CPU's experts need the GPU's attention and
router, and the GPU's next layer needs the experts' result. With hot experts, a split layer's VRAM experts (and the
shared expert) run on the GPU while the CPU computes the RAM experts: the graph puts the GPU's work first, and our
change to ggml's scheduler (Change 3 in [GGML_CHANGES.md](GGML_CHANGES.md)) lets the CPU start without waiting for
the GPU to finish and batches the copies between them. Measured: +22-44% on an RTX 3070 laptop with Qwen3.6-35B-A3B,
+12% on an RTX 5080 with hot experts, about the same on an RTX 5090 (its GPU work is already short).
`TIERLM_OVERLAP=0` turns it off.

## Verification

There is no llama.cpp to compare against, so every piece is checked against an independent reference:

| Piece | Reference | Result |
|---|---|---|
| Tokenizer | Hugging Face's OLMoE tokenizer, 14 tricky strings | All exact; all decode back byte-for-byte |
| Pre-tokenizer | The reference split regex, ~3,000 random and edge-case strings | 0 mismatches |
| Model graph | Plain NumPy reference of each architecture on tiny random models (`tests/tiny_models.py`: Llama, Llama 3 with RoPE factors, Mixtral, Qwen2, Qwen3, Qwen3 MoE, OLMoE) | ~2e-6 relative with an f32 cache, including flash attention, grouped-query attention, the KV cache and 300-token batched prompts |
| Images (Qwen3.5 vision) | Hugging Face's own Qwen3_5ForConditionalGeneration (PyTorch) on a tiny random model (`tests/tiny_vision.py`) | Image words ~7e-7 relative; a model reading images ~1e-4 (the same as its text-only difference) |
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

1. **Fewer, larger GPU steps.** On fast GPUs a token's time is mostly the ~2,400 small GPU kernels it launches, not
   the arithmetic: fused kernels for DeltaNet, the expert path and Flash-Next's hyper-connections.
2. **CPU and GPU across layers**: today they overlap within a split layer; the next layer's attention could start
   while the CPU finishes the last one's RAM experts.
3. **Cheaper expert moves** for the experts that follow the conversation: copies on their own GPU stream, overlapped
   with the next step.
4. **Faster CPU expert multiplies.**
5. **Usage-based precision** (rarely used experts stored at lower precision) and **expert-aware speculation**.
6. **More families**: Gemma 4, Phi, DeepSeek V4 and more real-model tests.
