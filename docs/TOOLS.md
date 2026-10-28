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
tierlm-server [model.gguf] [--port 8080] [--host 127.0.0.1] [-d auto|cpu|gpu|<device>] [-t N] [-c 4096]
              [--web web] [--models models] [--settings tierlm-settings.json]
```

- **Without a model**, the page opens a **setup screen**: it shows the GPUs and CPU it found with free memory
  and where the model will go (it picks the device itself: the best GPU the model fits on, else the CPU), the
  `.gguf` files in the models folder and its subfolders (marking which are supported), and the load settings,
  with more under **Advanced**. **Change setup** brings it back later.
- The models folder chosen on the page, and the settings used for each model, are saved in the `--settings`
  file. `--models` on the command line overrides the saved folder.
- It warns on the page if `VK_ICD_FILENAMES` or `VK_DRIVER_FILES` is set, since these can hide the GPU.
- **With a model**, it loads that model straight away (on the device given by `-d`) and opens the chat.
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
```

## tierlm-devices

Lists the backends compiled into this build, the devices ggml can use (with free memory), the version and the
build type. If a GPU is missing, it prints the error from the GPU backend, which `tierlm-run` cannot show. It
also warns if `VK_ICD_FILENAMES` or `VK_DRIVER_FILES` is set (and whether the file it names exists), since a
stale one can hide the GPU or crash Vulkan while it starts.

## tierlm-inspect

Summarises a GGUF model from its header only (fast, reads no weights).

```
tierlm-inspect model.gguf [--tensors]
```

Works on split models too (give it part 1). Shows the architecture, layer and expert counts, total size, size and share by role (experts, attention,
embedding, ...), and the per-layer size range. `--tensors` lists every tensor.

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
| `tierlm-logits model.gguf --gen[-spec] N id ...` | N tokens of greedy generation, without or with speculative decoding (prints token ids and, for `--gen-spec`, how many guesses were kept). `TIERLM_DRAFT_SCRIPT` / `TIERLM_DRAFT_CORRUPT` replace the draft layer with scripted, partly wrong guesses |

See [BUILD.md](BUILD.md#running-the-tests).

---

## Environment variables

| Variable | Effect |
|---|---|
| `TIERLM_NO_FA=1` | Use plain attention instead of flash attention |
| `TIERLM_KV_F32=1` | Keep the KV cache in f32 instead of f16 (exact; used by the tests) |
| `TIERLM_THREADS=N` | CPU threads for `tierlm-logits` and `tierlm-bench` |
| `TIERLM_PROFILE=1` | `tierlm-run`: per-token time breakdown and number of graph splits |
| `TIERLM_DUMP_SCHED=1` | Debugging: prints each point where the work switches between CPU and GPU |
| `TIERLM_CPU_OPS=MUL_MAT,...` | Debugging: runs the listed ggml operation types on the CPU, to find which GPU kernel differs |
| `TIERLM_DEVICE=cpu` | `tierlm-logits` (tests): device to load the model on |
| `TIERLM_KV_TYPE=q8_0` | KV cache precision for the tools and tests: `f16`, `q8_0`, `q4_0` or `f32` |
| `GGML_VK_VISIBLE_DEVICES=0` | Choose which Vulkan GPU(s) ggml uses |
| `VK_LOADER_DEBUG=error,warn` | Make the Vulkan loader explain driver problems |

On Windows `cmd`, set and clear them like this: `set TIERLM_NO_FA=1` then `set TIERLM_NO_FA=`.
