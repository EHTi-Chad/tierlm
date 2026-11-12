# Tools

All tools are built into `build/` (`build\` on Windows, with an `.exe` suffix). Examples use the OLMoE test
model; replace the path with your own.

Common options:

| Option | Meaning |
|---|---|
| `-d auto\|cpu\|gpu` | Device. `auto` (default) uses a GPU if the build has one and one is found, else the CPU. `gpu` fails if there is no GPU |
| `-t N` | CPU threads, for whatever runs on the CPU. Default: half the hardware threads (see [BUILD.md](BUILD.md#wsl-notes) for why) |
| `-c N` | Context window in tokens (KV cache size) |

---

## tierlm-server

Web chat and live dashboard, plus the HTTP API described in [API.md](API.md).

```
tierlm-server [model] [--port 8080] [--host 127.0.0.1] [--api-key KEY] [--no-web]
              [--web web] [--models models] [--settings tierlm-settings.json]
              [--saved] [-c 4096] [-t N] [-d auto|cpu|gpu|<device>] [--experts auto|layers|ram|busiest|follow]
              [--guesses N] [--lookup N] [--kv 32|16|8|5|4] [--batch N] [--connections N] [--vram-limit MiB]
              [--no-ssd] [--no-flash-attn] [--one-gpu] [--experts-used N] [--no-thinking] [--effort low|medium|high]
```

### From a console, with your own front end

Give the model and its settings on the command line and tierlm loads it at once and serves the API, with no
setup screen to click through. Any front end that speaks OpenAI's API (Open WebUI, SillyTavern, a coding agent,
your own script) connects to `http://<host>:<port>/v1`; `/v1/messages` and `/v1/responses` are there too.

```
tierlm-server.exe C:\models\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf --ctx 32768 --experts follow --guesses 2 --threads 6 --port 8040 --no-web
```

- **The model** is a `.gguf` file, or its name in the models folder as the page lists it (`unsloth/model.gguf`,
  `@tierlm/folder/model.gguf`). For a model in several files, give the first.
- **`--saved`** starts from the settings saved for that model: what the page last loaded it with, or what Find the
  fastest settings found. So you can tune once on the page and start it from a console or a script after that:
  `tierlm-server.exe model.gguf --saved --no-web`. Flags given beside `--saved` win over the saved ones.
- **`--no-web`** leaves the chat page out: the address answers with one line of text and the API is all there is.
  Without it the page is there as well, opened on the loaded model.
- **`--experts`** is where a MoE model's experts go when it doesn't fit in VRAM: `layers` (whole layers in VRAM, the
  rest in RAM), `ram` (experts in RAM, everything else in VRAM), `busiest` (the busiest experts in VRAM too),
  `follow` (the VRAM experts follow the conversation), or `auto`.
- **`--guesses N`** turns speculative decoding on with the model's draft block (N guesses ahead; 0 off);
  **`--lookup N`** guesses from the conversation's own text (0 off, -1 automatic).
- **`--kv`** is the context memory precision in bits; **`--connections N`** lets N replies be written at once, each
  with its own context of `--ctx` tokens.
- `--host 0.0.0.0` lets other devices on your network connect; give `--api-key` with it.
- A mistake in the options, or a model that can't be loaded, is said in the window and tierlm stops (exit code 1).
  `tierlm-server --help` lists everything.
- Loading this way doesn't change the saved settings.

- **Without a model**, the page opens a **setup screen**: it shows the GPUs and CPU it found with free memory
  and where the model will go (it picks the device itself: the best GPU the model fits on, else the CPU), the
  `.gguf` files in the models folder and its subfolders (marking which are supported), and the load settings,
  with more under **Advanced**. **Change setup** brings it back later.
- The models folder chosen on the page, and the settings used for each model, are saved in the `--settings`
  file. `--models` on the command line overrides the saved folder.
- It warns on the page if `VK_ICD_FILENAMES` or `VK_DRIVER_FILES` is set, since these can hide the GPU.
- **With a model**, it loads that model straight away with the settings given (see above) and opens the chat.
- `-d <device>` takes a name from `tierlm-devices`, such as `Vulkan0` or `CUDA0`. `auto` picks the dedicated
  GPU with the most memory, then an integrated GPU, then the CPU.
- Open `http://127.0.0.1:<port>` in a browser. Run it from the tierlm folder, or pass `--web <folder>` to
  point at the folder holding `index.html`. The page is re-read on every load, so you can edit
  `web/index.html` and just refresh.
- `--host 0.0.0.0` makes it reachable from other machines on your network. **There is no authentication**, so
  only do this on a network you trust.
- One reply (or model load) runs at a time; a second request meanwhile gets HTTP 409.
- Conversations reuse the KV cache: when you send the next message, only the new part is processed.

### The dashboard

| Panel | Shows |
|---|---|
| Generation speed | Current tok/s, and a speed-vs-context chart: one dot per generated token, x = tokens of context so far, y = speed |
| Context window | How much of the KV cache is in use |
| Where the model lives | Which tier (VRAM/RAM/SSD) holds the weights and KV cache, and device memory |
| Expert activity map | One cell per expert (rows = layers, columns = experts). Cells flash when the router picks them for the current token; in Heat mode, colour shows how often each has been used this session. Hover for counts |
| Hot-expert cache preview | If each layer could keep only N experts in VRAM, how often the router's pick would already be there, using the N most-used experts versus N at random |

## tierlm-run

Generate text from a prompt, greedily (always the most likely token).

```
tierlm-run model.gguf "prompt" [-n 32] [-t N] [-c 2048] [-d auto|cpu|gpu|<device>] [--split layers|experts] [--vram MiB]
```

Prints the device, where the model was placed, the top 5 candidates for the first token, the generated text,
and prompt/generation speeds. The prompt is used as-is, with the model's start token if it wants one (no chat
template); use the server for chat.

- `--split` chooses how a model bigger than VRAM is split (see [Splitting models](#splitting-models-between-vram-and-ram)).
- `--vram N` limits the VRAM tierlm may use to N MiB, for example to see how a model would run on a smaller GPU.
- `TIERLM_PROFILE=1` adds a per-token time breakdown and the number of GPU<->CPU hand-offs (`graph splits`).

## Splitting models between VRAM and RAM

When a model doesn't fit in the GPU's memory, tierlm's tier loader splits it. You don't need to do anything:
the setup screen shows the planned split before loading, and the dashboard shows where everything went.

| Mode | What goes in VRAM | GPU<->CPU hand-offs per token | Best for |
|---|---|---|---|
| `layers` (default) | The last layers that fit, whole, with their part of the KV cache, plus the output layer | 1 | Almost always, especially on Windows |
| `experts` | Attention, router and KV cache of every layer, then expert layers while they fit | 2 per layer in RAM | Very long conversations, where attention dominates |

In `layers` mode, every operation of a RAM layer is pinned to the CPU. Otherwise ggml would send small
weight-free operations (norms, attention over the cache) to the GPU and back, which cost more than they saved.

Measured on a laptop RTX 3070 (8 GB) with a Ryzen 9 5900HX and Qwen3-30B-A3B Q3_K_M (14.7 GB), 100 tokens:

| Setup | Generation |
|---|---|
| CPU only | 19.3 tok/s |
| 22 of 48 layers in VRAM (`layers`) | 26.3 tok/s (27.1 with `-t 6`) |

On laptops the CPU and GPU share a power budget, so the CPU part runs a little slower while the GPU is busy;
desktops gain more.

## tierlm-bench

Generation speed at several context lengths: for each, fill the cache with that many tokens, then time 32
single-token steps.

```
tierlm-bench model.gguf [-d auto|cpu|gpu] [-t N] [contexts...]       # default contexts: 64 512 1024 2048
tierlm-bench model.gguf --parallel 1,2,4,8 512                        # several conversations at once, 512 tokens each
```

With `--parallel`, each number of conversations is advanced one token each per step, and the total and
per-conversation speed printed.

## tierlm-devices

Lists the backends compiled into this build, the devices ggml can use (with free memory), the version and the
build type. If a GPU is missing, it prints the error from the GPU backend, which `tierlm-run` cannot show. It
also warns if `VK_ICD_FILENAMES` or `VK_DRIVER_FILES` is set (and whether the file it names exists), since a
stale one can hide the GPU or crash Vulkan while it starts.

## tierlm-inspect

Summarises a GGUF model from its header only (fast, reads no weights).

```
tierlm-inspect model.gguf [--tensors | --all]
```

Works on split models too (give it part 1). Shows the architecture, layer and expert counts, total size, size and share by role (experts, attention,
embedding, ...), and the per-layer size range. `--tensors` lists every tensor. `--all` prints every setting in the
file and every tensor with its type and shape (`[ne0, ne1, ne2, ne3]`, ggml's order): what to send when a new kind of
model doesn't load.

## tierlm-plan

Dry run of the VRAM/RAM/SSD placement planner for a memory budget.

```
tierlm-plan model.gguf [--vram GiB] [--ram GiB]
```

Experts are split into individual units (one expert in one layer). Small, always-used parts (norms, router,
attention, output) are placed first; the embedding table goes to RAM; experts fill the remaining VRAM, then
RAM, then SSD. It prints GiB per role and tier, experts in VRAM per layer, and the expected expert reads per
token from each tier assuming even routing. This is a plan only: the tier loader that acts on it is not built
yet.

## tierlm-embed

The embedding of a text from an embedding model (`bert`, `nomic-bert`), for scripts and tests:

```
tierlm-embed nomic-embed-text-v1.5.Q8_0.gguf "search_query: what is tierlm?"   # token ids, then the vector
tierlm-embed model.gguf --ids 101,2023,102 -d auto                            # exactly these token ids, on the GPU if it fits
```

It prints the token ids and the pooled embedding (not normalised; `/v1/embeddings` normalises).

## tierlm-vision

Image reading, for tests and for trying a vision (mmproj) file by hand:

```
tierlm-vision mmproj-Qwen3.5-9B-F16.gguf page.png words.f32            # the image words (prints rows cols width)
tierlm-vision mmproj.gguf --patches page.png patches.f32                # just the resized image, cut into patches
tierlm-vision mmproj.gguf --chat model.gguf page.png logits.f32 all 5 17 -1 9   # a model reading it (-1: the image)
```

`TIERLM_DEVICE=cpu` (or `CUDA0`, ...) picks the device; `TIERLM_VISION_EXACT=1` uses plain attention even for big
images (tests).

## tierlm-kv

Dumps all metadata keys in a GGUF file (architecture settings, tokenizer type, vocabulary sizes, chat
template...).

## Test harnesses

| Tool | Used by |
|---|---|
| `tierlm-tok-test model.gguf tests/tok_cases.txt` | Tokenizer test: compares token IDs with Hugging Face's, and checks decoding round-trips |
| `tierlm-logits model.gguf [--step] id ...` | Prints last-position logits for token IDs; `--step` feeds one token per call through the KV cache. Used by `tests/tiny_models.py` |
| `tierlm-logits model.gguf --mtp[-step] id ...` | The MTP draft layer's prediction after the ids (all at once, or one position at a time). Used by `tests/tiny_qwen35.py` |
| `tierlm-logits model.gguf --rollback B KEEP id ...` | Evaluates the last B+1 ids as an undoable check, keeps KEEP of the B, then the last id: must match plain evaluation |
| `python tests/tiny_flashnext.py build/tierlm-logits` | Qwen3.8 Flash-Next (`qwen4exp`): tiny random models built with transformers' own Flash-Next code, including long inputs past the sparse-attention budget, compared logit by logit whole, one token at a time and saved/restored (needs `pip install torch transformers gguf`) |
| `python tests/tiny_deepseek.py build/tierlm-logits` | DeepSeek V2 / V3 / Kimi (`deepseek2`): tiny random models built with transformers' own DeepSeek code, written as GGUF in all three layouts real files use, compared logit by logit (needs `pip install torch transformers gguf`) |
| `tierlm-logits model.gguf --gen[-spec] N id ...` | N tokens of greedy generation, without or with speculative decoding (prints token ids and, for `--gen-spec`, how many guesses were kept). `TIERLM_DRAFT_SCRIPT` / `TIERLM_DRAFT_CORRUPT` replace the draft layer with scripted, partly wrong guesses |

See [BUILD.md](BUILD.md#running-the-tests).

---

## Environment variables

| Variable | Effect |
|---|---|
| `TIERLM_NO_FA=1` | Use plain attention instead of flash attention |
| `TIERLM_KV_F32=1` | Keep the KV cache in f32 instead of f16 (exact; used by the tests) |
| `TIERLM_SSD=0` | Never use the SSD tier: everything that isn't in VRAM goes in RAM, as before 0.8.2 |
| `TIERLM_SSD_LAYERS=N` | Tests: put exactly N groups of weights on the SSD tier (the least-used expert layers first), whatever fits |
| `TIERLM_SSD_DIR=folder` | Where the SSD tier's copy goes when none is chosen in the setup screen (tests) |
| `TIERLM_SSD_PREFETCH=0` | SSD tier: don't ask the drive for the chosen experts ahead of time (to measure what read-ahead gains) |
| `TIERLM_SSD_LOG=1` | SSD tier: every 2 seconds, how much has been asked for ahead from the drive so far |
| `TIERLM_CUDA_CHECK=0` | Don't check which NVIDIA cards the CUDA engine covers (normally a card it has no code for is hidden from CUDA and used through Vulkan instead; not done either when `CUDA_VISIBLE_DEVICES` is set) |
| `TIERLM_TRACE=1` | Debugging: prints a summary (size, sum, RMS, first numbers) of each layer's main results as they're computed, to compare a model step by step with a reference |
| `TIERLM_THREADS=N` | CPU threads for `tierlm-logits` and `tierlm-bench` |
| `TIERLM_PROFILE=1` | Timing: `tierlm-run` prints a per-token breakdown and the number of graph splits; every tool and the server also print, every 64 reply steps, each device's input-copy and work time per step (`profile:` lines). Each split is waited for while it's on, so everything runs a little slower |
| `TIERLM_OVERLAP=0` | Turns off CPU and GPU working at once on layers split between VRAM and RAM (and the batched copies between them). `2`: the same order of work, but every step waits for the one before (a test: the output must not change) |
| `TIERLM_PROMPT_CACHE_GB=n` / `TIERLM_PROMPT_CACHE_MIN=n` | `tierlm-server`: the size limit of the openings saved on the drive (`tierlm-cache/`; default 8, 0 turns it off; the setting is `prompt_cache_gb`), and the shortest system message saved, in tokens (default 1,024) |
| `TIERLM_ANCHORS=12` / `TIERLM_ANCHORS_MB=1536` / `TIERLM_ANCHOR_EVERY=1024` | Hybrid models: how many checkpoints (points an edited conversation can go back to) are kept, their RAM limit, and the fewest tokens between two taken while reading a prompt (`TIERLM_ANCHOR_EVERY`: `tierlm-server`) |
| `TIERLM_REUSE=0` | Builds the graph for every reply token instead of building it once and running it again (about 3 ms a token) |
| `TIERLM_USAGE=<file>` | `tierlm-run`: read expert usage counts saved by the app (`tierlm-usage/*.bin`), so the busiest experts go to VRAM as in the app; for speed tests outside it |
| `TIERLM_VRAM_SCRATCH_MB=n` | VRAM kept free beyond the measured compute buffers, for the GPU engine's scratch memory and the driver (default: 1/32 of the card, 192 to 640: 256 on an 8 GB card, 640 on 20 GB and up). tierlm measures the compute buffers after loading, and if VRAM is short it loads again once with more kept free, and remembers it per model (`tierlm-usage/*.vram`), because on Windows VRAM that runs out doesn't fail, the driver quietly moves memory to system RAM and the GPU slows down many times over |
| `TIERLM_COLD_MAPPED=n` | Tests: read the RAM part of the n least-used split expert layers straight from the model file instead of RAM (`0`: never). By default only when the weights can't fit in RAM at all (more than the PC's RAM less 3 GB): held in RAM they run much faster (Flash-Next on a 64 GB PC: about 50 tok/s, against 31 read from the file) |
| `TIERLM_BATCH_LOG=1` | Server, Connections above one: every 2 seconds, how many replies shared each model step, the compute time per step and the share of the time spent computing (`batcher:` lines). Close to the number of replies running means they share steps as they should |
| `TIERLM_DUMP_SCHED=1` | Debugging: prints each point where the work switches between CPU and GPU |
| `TIERLM_CPU_OPS=MUL_MAT,...` | Debugging: runs the listed ggml operation types on the CPU, to find which GPU kernel differs |
| `TIERLM_DEVICE=cpu` | `tierlm-logits` (tests): device to load the model on |
| `TIERLM_KV_TYPE=q8_0` | KV cache precision for the tools and tests: `f16`, `q8_0`, `q5_0`, `q4_0` or `f32` |
| `TIERLM_NO_REPACK=1` | Keep RAM weights in the file's layout instead of rearranging them for the CPU |
| `TIERLM_RPC=host:port[,host:port]` | Tests: add `tierlm-rpc-server` instances as stand-in GPUs (needs a build with `-DGGML_RPC=ON`) |
| `TIERLM_VRAM_MB=N` / `TIERLM_VRAM_BYTES=n[,n...]` | Tests: VRAM budget for `tierlm-logits`, in MiB, or in bytes per GPU |
| `TIERLM_SPLIT=layers\|experts` | Tests: split mode for `tierlm-logits` |
| `TIERLM_LOAD_TRACE=1` | at load, where the time went: reading the file, splitting experts between VRAM and RAM, copying the rest |
| `TIERLM_SHOW_PROMPT=1` | tierlm-server: write each prompt to the window as the model reads it (chat format, tools, thinking), to compare with a model's own template |
| `TIERLM_MAMBA_UNROLL=N` | Nemotron: write the Mamba recurrence out as ordinary operations for batches of up to N tokens, in place of ggml's scan (default: only for checks of guesses on GPUs other than CUDA, whose scan keeps no undo points; 0: never) |
| `TIERLM_EXPERT_USAGE=seed` | Tests: made-up expert usage counts, so models load with hot experts (some layers whole, some split) |
| `TIERLM_SHOW_PLACEMENT=1` | `tierlm-logits`: print where the model was placed (per GPU, and the hot-expert count), and with live hot experts the picks served from VRAM and the swaps |
| `TIERLM_LIVE_EXPERTS=1` | Load with live hot experts (as `live_experts` in `/api/load`); needs hot experts (usage counts) |
| `TIERLM_TRACE_CACHE=1` | `tierlm-server`: for each request, how much of the prompt cache it reuses and where it stops matching |
| `TIERLM_LIVE_SHARE=0.08` | Live hot experts: the share of each step's time that moving experts may take |
| `TIERLM_LIVE_MB=32` / `TIERLM_LIVE_PER_LAYER=2` | Live hot experts: most MB moved per step (both ways), most swaps per layer per step |
| `TIERLM_LIVE_PROMPT_MS=250` / `TIERLM_LIVE_PROMPT_HIT=0.62` | Live hot experts: the longest burst of moves after a prompt batch the VRAM experts served poorly (0: never), and the share of the batch's picks served from VRAM under which it starts (full length 17 points lower) |
| `TIERLM_LIVE_TRACE=1` | Live hot experts: prints, for each prompt batch, the share of its picks the VRAM experts served and what was moved |
| `TIERLM_COLD_OFFLOAD_MIN=n` | Experts split between VRAM and RAM: the prompt batch size from which the GPU does the RAM experts' work too, copying them over for the batch (default 512 on a 24 GB card or more, 128 otherwise; ggml's rule, and tierlm's before 0.8.7, is 32) |
| `TIERLM_LIVE_MARGIN=1.3` | Live hot experts: how much more an expert in RAM must be wanted than one in VRAM to replace it (times its score, + 0.5) |
| `TIERLM_LIVE_SYNC=1` | Live hot experts: wait for each copy instead of queuing them all on the GPU (for comparison) |
| `TIERLM_NO_TIMES=1` | `tierlm-server`: no time (HH:MM:SS) in front of each line in its window (the page's console still shows it) |
| `TIERLM_SPEC_SAMPLING=0` | `tierlm-server`: at temperature > 0, guess the draft block's best word and keep it only when the model's own draw matches, instead of speculative sampling (for comparison; the replies' odds are the model's either way) |
| `TIERLM_MTP_VRAM=0` | Don't put the draft block's experts in VRAM before the model's own (the default since 0.8.7, so guessing runs wholly on the GPU): they compete with the model's experts by use, as before |
| `TIERLM_PLANS=0` | Models with a draft block: build and place the check's graph and the draft block's graph every time, on the one scheduler, as before 0.8.7 (see "Kept plans" in ARCHITECTURE.md). `1`: keep them even when the check's working memory is more than 1/32 of the card |
| `TIERLM_PLAN_WIDTH=n` | The widest check (guesses + 1) the kept check plan reserves working memory for at load (default 5, for 4 guesses; a wider check makes it grow) |
| `TIERLM_NO_NGRAM_PREFETCH=1` | Flash-Next with its n-gram table on the SSD tier: read the rows one by one as the graph needs them, not all at once before each step (for comparison) |
| `TIERLM_KEEP_RAM=0` | Don't hold the weights in system RAM resident after loading (Windows: a raised working-set minimum) |
| `TIERLM_SAMPLE_RUNS=8000` | `tierlm-logits --gen` / `--gen-spec`: draw that many separate replies (with `TIERLM_SAMPLE_TEMP`, `TIERLM_SAMPLE_TOPK`), one per line, for the speculative-sampling test |
| `TIERLM_LIVE_EAGER=1` | Tests: live hot experts swap at every chance (no margin), to exercise swapping |
| `TIERLM_TEST_FRACS="0.4 0.6"` | `tests/tiny_models.py`: the VRAM sizes (fractions of the model) for split tests |
| `GGML_VK_VISIBLE_DEVICES=0` | Choose which Vulkan GPU(s) ggml uses |
| `VK_LOADER_DEBUG=error,warn` | Make the Vulkan loader explain driver problems |

On Windows `cmd`, set and clear them like this: `set TIERLM_NO_FA=1` then `set TIERLM_NO_FA=`.
