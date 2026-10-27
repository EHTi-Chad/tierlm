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
tierlm-server model.gguf [--port 8080] [--host 127.0.0.1] [-d auto|cpu|gpu] [-t N] [-c 4096] [--web web]
```

- Open `http://127.0.0.1:<port>` in a browser. Run it from the project folder, or pass `--web <folder>` to
  point at the folder holding `index.html`. The page is re-read on every load, so you can edit
  `web/index.html` and just refresh.
- `--host 0.0.0.0` makes it reachable from other machines on your network. **There is no authentication**, so
  only do this on a network you trust.
- One reply is generated at a time; a second request while one is running gets HTTP 409.
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
tierlm-run model.gguf "prompt" [-n 32] [-t N] [-c 2048] [-d auto|cpu|gpu]
```

Prints the device, the top 5 candidates for the first token, the generated text, and prompt/generation
speeds. The prompt is used as-is (no chat template); use the server for chat.

## tierlm-bench

Generation speed at several context lengths: for each, fill the cache with that many tokens, then time 32
single-token steps.

```
tierlm-bench model.gguf [-d auto|cpu|gpu] [-t N] [contexts...]       # default contexts: 64 512 1024 2048
```

## tierlm-devices

Lists the backends compiled into this build, the devices ggml can use (with free memory), and the build type.
If a GPU is missing, it prints the error from the GPU backend, which `tierlm-run` cannot show.

## tierlm-inspect

Summarises a GGUF model from its header only (fast, reads no weights).

```
tierlm-inspect model.gguf [--tensors]
```

Shows the architecture, layer and expert counts, total size, size and share by role (experts, attention,
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
| `tierlm-logits model.gguf [--step] id ...` | Prints last-position logits for token IDs; `--step` feeds one token per call through the KV cache. Used by `tests/tiny_olmoe.py` |

See [BUILD.md](BUILD.md#running-the-tests).

---

## Environment variables

| Variable | Effect |
|---|---|
| `TIERLM_NO_FA=1` | Use plain attention instead of flash attention |
| `TIERLM_KV_F32=1` | Keep the KV cache in f32 instead of f16 (exact; used by the tests) |
| `TIERLM_THREADS=N` | CPU threads for `tierlm-logits` and `tierlm-bench` |
| `GGML_VK_VISIBLE_DEVICES=0` | Choose which Vulkan GPU(s) ggml uses |
| `VK_LOADER_DEBUG=error,warn` | Make the Vulkan loader explain driver problems |

On Windows `cmd`, set and clear them like this: `set TIERLM_NO_FA=1` then `set TIERLM_NO_FA=`.
