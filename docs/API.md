# Server API

`tierlm-server` serves the web page, an OpenAI-compatible API and its own small JSON API on the same port
(`http://127.0.0.1:8040` when started with `start-tierlm.bat`). Only this computer can connect unless you turn
on **Allow other devices**; see [Other devices and API keys](#other-devices-and-api-keys).

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | The web page (`web/index.html`) |
| GET | `/api/info` | Model, device and memory details (`{"loaded": false}` before a model is loaded) |
| GET | `/api/devices` | GPUs and CPU the engine can use, with free memory |
| GET | `/api/models` | `.gguf` files in the models folder (and subfolders), and whether each is supported |
| POST | `/api/folder` | Change the models folder (saved for next time) |
| POST | `/api/pick-folder` | Open a folder picker on the computer running tierlm (Windows) |
| POST | `/api/load` | Load a model, with load settings |
| POST | `/api/unload` | Free the loaded model |
| GET, POST | `/api/context` | What a different context length would take, and changing it without a reload |
| POST | `/api/chat` | Generate a reply, streamed as Server-Sent Events |
| POST | `/api/reset` | Clear the KV cache |
| GET | `/api/network` | Network access, API key status, and the addresses apps should use |
| POST | `/api/network` | Change network access or the API key (this computer only) |
| GET | `/v1/models` | OpenAI-compatible: models |
| POST | `/v1/chat/completions` | OpenAI-compatible: chat, streamed or not |
| GET | `/api/v0/models`, `/api/v0/models/<id>` | LM Studio's REST API: models with their type (`llm`, `vlm`, `embeddings`) and state |
| POST | `/api/v0/chat/completions`, `/api/v0/embeddings` | LM Studio's REST API: the same as the `/v1` ones |

The first part of this page is the OpenAI-compatible API, for other apps. The rest documents tierlm's own
`/api` endpoints, which the web page uses.

## OpenAI-compatible API

Apps that can talk to OpenAI can use tierlm instead: set the **base URL** to `http://127.0.0.1:8040/v1` (or
this computer's network address, see below) and any API key, unless you've set one in tierlm. Tested with
the official `openai` Python library; it follows the same format as LM Studio, Ollama and llama.cpp's server.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8040/v1", api_key="not-needed")
reply = client.chat.completions.create(
    model="Qwen3.6-35B-A3B-Q8_0.gguf",          # or any name: the loaded model is used
    messages=[{"role": "user", "content": "Explain RAID 6 in two sentences."}],
    stream=True)
for chunk in reply:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### GET /v1/models

The loaded model first (`"loaded": true`), then the loaded embedding model if there is one (`"type": "embeddings"`),
then every supported model in the models folder. `id` is the
model's path inside the folder, e.g. `lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf`.

### POST /v1/embeddings

Embeddings from the **embedding model**, a second model loaded next to the chat model (side menu → **Model info** →
**Embedding model**, or `POST /api/embed/load`). Embedding requests don't wait for a chat reply to finish: the two
models take turns on the GPU one step at a time.

```json
{"model": "any", "input": ["first text", "second text"], "encoding_format": "float", "dimensions": 256}
```

- `input`: a string, a list of strings, a list of token ids, or a list of such lists.
- `encoding_format`: `"float"` (default) or `"base64"` (little-endian float32, which the OpenAI SDKs ask for by default).
- `dimensions` (optional): keep the first n numbers, then normalise again, for models trained for it (nomic-embed-text v1.5:
  64 to 768).

Each vector is normalised to length 1, as OpenAI's are, so a dot product is the cosine similarity. Texts longer than
the model takes (2,048 tokens for nomic-embed-text, 512 for most BERT models) are cut to fit. The reply:
`{"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [...]}], "model": "...",
"usage": {"prompt_tokens": n, "total_tokens": n}}`. Without an embedding model loaded, the answer is a 400 error
saying how to load one.

Supported: BERT-style models (`bert`: bge, all-MiniLM, e5, ...) and Nomic's (`nomic-bert`: nomic-embed-text v1, v1.5).
nomic-embed-text expects a task prefix on each text, as its model card describes: `search_document: ` for what you
store and `search_query: ` for what you search with.

### POST /v1/chat/completions

| Field | Notes |
|---|---|
| `model` | From an app on this computer, an `id` from `/v1/models` loads that model first (with the settings last used for it in tierlm), which takes as long as loading it on the setup screen. Any other name, such as `gpt-4o`, uses the loaded model, and so does every name from another device (which model is loaded is a setting of the computer running tierlm). |
| `messages` | `system`, `developer` (same as system), `user`, `assistant`. Content can be a string or a list of parts: text, and images (see [Images](#images)). Also `tool` (a tool's result), and `assistant` messages with `tool_calls` and `reasoning_content`. |
| `tools`, `tool_choice` | OpenAI function tools. The model may answer with `tool_calls` instead of text (see Tool calling below). `tool_choice: "none"` leaves the tools out. |
| `stream` | `true` streams `chat.completion.chunk` events, ending with `data: [DONE]`. `stream_options: {"include_usage": true}` adds a final chunk with token counts. |
| `max_tokens`, `max_completion_tokens` | Longest reply. Left out: no limit (until the model finishes or the context is full). |
| `temperature`, `top_p`, `seed`, `stop` | As in OpenAI. `stop` is a string or a list of strings; the reply ends just before one. |
| `presence_penalty`, `frequency_penalty` | As in OpenAI: taken off the logit of a token used in the last 64 tokens (`repeat_last_n`), once, or once per use. Off (0) unless given. Qwen suggests `presence_penalty: 1.5` for its models with thinking off. |
| `top_k`, `min_p`, `repeat_penalty` | Extras, as in LM Studio and llama.cpp. `repeat_penalty` is off (1.0) unless given, because it harms code. |

The reply includes `usage` (`prompt_tokens`, `completion_tokens`). `finish_reason` is `stop` (the model
finished, or a stop string), `tool_calls` (the model wants tools run), or `length` (`max_tokens`, or the context
window is full).

#### Tool calling

For coding agents such as opencode, Cline, Continue's agent mode and Cursor. tierlm describes the tools to the model
in the format it was trained on (Qwen3.5 / 3.6 and Qwen3-Coder write `<tool_call><function=...>` blocks; Qwen3,
Qwen2.5 and other models the JSON form `<tool_call>{"name": ..., "arguments": ...}</tool_call>`; gpt-oss its harmony
format, a message on the `commentary` channel addressed `to=functions.NAME`) and reads its calls
back out as OpenAI `tool_calls`, with `arguments` as a JSON string, and `finish_reason: "tool_calls"`. Several calls
in one reply are returned together. tierlm never runs a tool itself: the app does, and sends the result back as a
`tool` message.

A thinking model's reasoning is sent apart from the answer, as `reasoning_content` (in the message, or in the
stream's deltas), the way LM Studio, llama.cpp and DeepSeek do, so an app that shows `content` shows just the answer;
a reply cut off by `max_tokens` while still thinking has an empty `content`. An app that sends the reasoning back keeps
the model's memory of the conversation exact. Streamed, the answer text arrives as it is written and tool calls
arrive whole in one chunk at the end.

**Thinking on or off.** For models whose chat template can switch it (Qwen3 and later), the page's Settings set it for
the model (saved with its settings), and that applies to apps too. A request can choose for itself with
`"chat_template_kwargs": {"enable_thinking": false}` (as for vLLM and llama.cpp), `"enable_thinking": false`, or
`"think": false` (as for Ollama). gpt-oss always reasons; `"reasoning_effort": "low" | "medium" | "high"` sets how
much (`"none"` turns thinking off for the other models). `GET /api/info` says `think_switch`, `thinking` and
`reasoning_effort`; `POST /api/thinking {"thinking": false}` or `{"effort": "low"}` sets the model's default (on this
computer only).

Coding agents send long prompts (opencode's instructions and tool list are about 11,000 tokens): load the model
with a context of at least 32K. The first request reads all of it; later ones reuse it from the cache.

One reply is generated at a time. Requests (from apps and from the web page) wait their turn in line, first come
first served, and a request whose app disconnects while waiting leaves the line. A request gives up after waiting 30
minutes (`503`). Requests share the model's memory of the conversation, and up to 8 conversations are kept aside, so
an app that resends a growing conversation only pays for the new part, even when others took turns in between.

#### Images

Models that read images (Qwen3.5 / 3.6, with their `mmproj-*.gguf` vision file in the same folder; `/v1/models`
shows them as `"type": "vlm"` with `"capabilities": ["vision"]`) take images the way OpenAI's API does:

```json
{"role": "user", "content": [
  {"type": "text", "text": "What does this receipt say?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}
]}
```

- Send images as base64 `data:` URLs (PNG, JPEG, GIF or BMP); tierlm doesn't fetch `http(s)` addresses.
  `{"type": "input_image", "image_url": "data:..."}` (OpenAI's newer form) works too.
- An image becomes one "image word" per 32 x 32 pixels, at most 2,048 (bigger images are scaled down, keeping
  their shape; a full scanned page at that size is about 1,250 x 1,650 pixels), or 1,024 when only the CPU reads
  images. `"image_tokens": 512` (any number from 64) in the request sets another limit. These count towards the
  context.
- The same image sent again (each turn of a conversation sends it again) is recognised and not read twice, and
  the conversation's cache is reused as for text.
- With images in a conversation, speculative decoding is off for that reply (the draft block reads text only).

Not yet: `logprobs`; `n` > 1; embeddings.

Errors use OpenAI's format: `{"error": {"message": "...", "type": "...", "code": ...}}`.

### LM Studio's REST API (`/api/v0`)

For apps written for LM Studio, which list models at LM Studio's own address:

- `GET /api/v0/models`: every model, in LM Studio's fields: `id`, `type` (`llm`; `vlm` for one that reads images, which a
  model with its `mmproj` file next to it does; `embeddings`), `state` (`loaded` or `not-loaded`), `publisher` (the first
  folder of the id), `arch`, `compatibility_type` (`gguf`), `quantization` (from the file name), `max_context_length`,
  and `loaded_context_length` for a loaded model.
- `GET /api/v0/models/<id>`: one of them (404 if there is no such model).
- `POST /api/v0/chat/completions` and `POST /api/v0/embeddings`: the same as `/v1/chat/completions` and `/v1/embeddings`.

These need the API key when one is set, like `/v1`.

### Other devices and API keys

By default tierlm only answers this computer. In the web page, **Connect other apps** has:

- **Allow other devices**: tierlm listens on your network too, and shows the address to use, such as
  `http://192.168.1.20:8040/v1`. Needed for other PCs and phones, for Open WebUI running in Docker (Docker's
  traffic doesn't count as this computer), and for tunnels. It takes effect at once.
- **API key**: when set, every `/v1` request must send it (`Authorization: Bearer <key>`, which is what apps do
  with the "API key" you give them), and so must everything else that doesn't come from this computer; the web
  page asks for it once on other devices. Requests passed on by a tunnel or proxy (they carry headers such as
  `X-Forwarded-For` or `Cf-Connecting-Ip`) never count as this computer.

Both are saved in `tierlm-settings.json`. `--host 0.0.0.0` and `--api-key KEY` on the command line override
them.

**Other devices use the model; they don't run the computer.** From anywhere but the computer running tierlm, the
server answers chats (`/api/chat`, `/v1`, `/api/v0`) and what the page needs to show them (`/api/info`, speed and
context), and refuses with 403 everything that changes the computer: loading or unloading models (also through a
`model` name in a request), settings (`/api/spec`, `/api/network`), Find the fastest settings and the benchmark, the
models folder, the embedding model and its download, the workspace and saved chats; and the console, hardware and
model-file lists. The page there shows the chat, speed, context and the address to connect with. `/api/info` adds
`"remote": true` there, and gives the model's file name instead of its full path.

**What another device can and can't reach** (checked again for 0.8.7):

- It can chat with the loaded model, and see the model's name, context size, speed, whether an embedding model is
  loaded, and tierlm's own network address.
- It can't read or write any file on the computer: not the workspace (it isn't even told which folder is chosen;
  until 0.8.7 `/api/workspace`, and a chat request asking for the workspace tools, gave the folder's path away),
  not saved chats, not the models folder, not the console, not the settings file with the API key.
- It can't change anything: models, settings, network access and the key stay with the computer running tierlm.
- A request body over 256 MB is refused (413), so one request can't fill the computer's memory (since 0.8.7).
- The connection is plain HTTP: on a network you don't trust, anyone who can watch the traffic sees the chats and
  the API key. Use it on your own network, or put a tunnel or reverse proxy with HTTPS in front.
- Without an API key, anyone who can reach the address can use the model. Set one whenever other devices are allowed.

## tierlm's own API (`/api`)

### GET /api/info

Example (values depend on the model and machine):

```json
{
  "name": "OLMoE 1B 7B 0924 Instruct", "arch": "olmoe",
  "layers": 16, "experts": 64, "experts_used": 8, "vocab": 50304,
  "ctx": 4096, "n_past": 0,
  "device": "Vulkan0 (Intel(R) Arc(TM) 140V GPU (16GB))", "on_gpu": true, "threads": 4,
  "weights_bytes": 4209221632, "kv_bytes": 536870912,
  "device_free": 12884901888, "device_total": 19219415040,
  "kv_type": "f16", "flash_attn": true, "batch": 256, "experts_default": 8, "seed": -1,
  "version": "0.5.0"
}
```

It also reports where the model lives: `vram_weights` and `ram_weights` (bytes), `split_mode`, `gpus` (one entry per
GPU in use: `name`, `weights`, `kv`, `layers`), `ffn_vram` (per layer: are its experts in VRAM), and
`speculative`, `draft_max`, `has_draft_block` and `mtp_loaded` (the draft layer is in memory, so `/api/spec` can
switch speculation on), `lookup` (most tokens prompt lookup guesses now, 0 when off) and `lookup_setting` (as set: -1 automatic), `hot_experts` (experts placed in VRAM by usage), `hot` (per layer: those experts' numbers, which change as you chat with
live hot experts), `live_experts` (on or off) and `live` (`picks`, `hits`: picks served from VRAM, `swaps`, `swap_ms`),
`hot_share` (the share of the learned expert use those VRAM experts cover, 0 to 1) and `learned_tokens` (about how
many tokens of use the expert counts come from), and for images `vision` (the model reads them: its mmproj file is
loaded), `vision_file`, `vision_device` and `image_tokens` (most image words per image).

While a reply is streaming or a model is loading, `/api/info` returns the last known details with
`"busy": true` and `waiting` (requests in line) instead of waiting. It also gives `ram_free` (system RAM free, bytes) and `tuned_tps`
(what **Find the fastest settings** measured for this model on this PC, 0 if never run): the page uses them to say
when a reply was much slower than that, and why it probably was.
Since 0.8.7 also `prompt_cache` (`files`, `bytes`, `limit_bytes`, `min_tokens`: the openings saved on the drive, see
ARCHITECTURE.md) and `ram_short` (true when the model was loaded with less RAM free than it keeps there, so the system
will page and replies will be slow), with `ram_need_at_load` and `ram_free_at_load` in bytes.

### GET /api/devices

```json
{"devices": [
   {"id": "Vulkan0", "description": "Intel(R) Arc(TM) 140V GPU (16GB)", "type": "igpu", "backend": "Vulkan",
    "free": 17179869184, "total": 19219415040},
   {"id": "CPU", "description": "Intel(R) Core(TM) Ultra 7 258V", "type": "cpu", "backend": "CPU",
    "free": 11811160064, "total": 33822867456}],
 "threads": 4, "hw_threads": 8, "version": "0.5.0", "warnings": []}
```

`type` is `gpu` (dedicated), `igpu` (integrated, shares system memory) or `cpu`. Pass `id` as the `device`
when loading.

`warnings` lists problems found on this computer, each with a `code` and a `message`: `vk_env` when
`VK_ICD_FILENAMES` or `VK_DRIVER_FILES` is set (these can hide the GPU; `missing` is true if the file they
name doesn't exist), and `no_gpu` when a GPU build finds no GPU.

### GET /api/models

```json
{"folder": "C:\\tierlm\\models", "exists": true, "recursive": true, "default_ctx": 4096, "current": "",
 "can_pick": true, "last_model": "", "saved": {},
 "suggestions": [{"name": "LM Studio models", "folder": "C:\\Users\\you\\.lmstudio\\models"}],
 "models": [
   {"id": "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", "folder": "", "file": "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", "name": "OLMoE 1B 7B 0924 Instruct", "arch": "olmoe",
    "supported": true, "bytes": 4212186208, "layers": 16, "experts": 64, "experts_used": 8,
    "ctx_train": 4096, "kv_bytes_per_token": 131072},
   {"id": "meta/Llama-3.2-1B-Q4_K_M.gguf", "folder": "meta", "file": "Llama-3.2-1B-Q4_K_M.gguf", "arch": "llama", "supported": false,
    "reason": "the llama architecture is not supported yet", "bytes": 807694464}]}
```

Only the file headers are read, so this is fast. `kv_bytes_per_token` is the KV cache cost per token of
context (f16), useful for estimating memory: model bytes + `kv_bytes_per_token` × context.

`id` is the model's path relative to the folder; pass it to `/api/load`. A model split over several files is
listed once, by its first part, with `"parts": N` and the total size of all parts; parts 2..N are not listed. With `recursive` on, subfolders are
searched too (up to 4 levels, as LM Studio stores models as `publisher/model/file.gguf`). `suggestions` lists
other model folders found on this computer. `saved` holds the settings last used for each model, by `id`.

### POST /api/folder

```json
{"folder": "D:\\AI\\models", "recursive": true}
```

Either field can be left out. The choice is saved in `tierlm-settings.json` (in the folder tierlm runs from)
and used next time, unless `--models` is given on the command line.

### POST /api/pick-folder

Opens the Windows folder picker on the computer running tierlm and returns
`{"cancelled": false, "folder": "D:\\AI\\models"}`; it doesn't change the folder by itself. Only answers
requests from the same computer. Elsewhere it returns HTTP 501.

### POST /api/load

```json
{"model": "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", "device": "auto", "ctx": 4096, "threads": 4,
 "flash_attn": true, "kv_type": "f16", "batch": 256, "experts_used": 0, "seed": -1}
```

`model` is an `id` from `/api/models` (paths outside the models folder are refused). `device` is `auto`,
`cpu`, `gpu`, or an `id` from `/api/devices`. Everything except `model` is optional:

| Field | Meaning |
|---|---|
| `ctx` | Context length in tokens |
| `threads` | CPU threads |
| `flash_attn` | Flash attention on/off |
| `kv_type` | KV cache precision: `f16` (default), `q8_0` (8-bit, half the memory), `q5_0` (5-bit, a third; since 0.8.7), `q4_0` (4-bit, a quarter, lower quality) or `f32` (exact). 8-bit and below turn on flash attention |
| `batch` | Tokens per step when reading a prompt (default 256) |
| `experts_used` | Active experts per token for MoE models; 0 keeps the model's own value |
| `seed` | -1 for random sampling; any other number makes replies repeatable |
| `speculative` | `true` loads the model's MTP draft layer and uses speculative decoding (when the file, or an `mtp-*.gguf` next to it, has one: `"mtp": true` in `/api/models`) |
| `draft_max` | Most tokens the draft layer guesses per check, 1 to 8 (default: 2 for MoE models, 3 for dense) |
| `lookup` | Prompt lookup: most tokens guessed per check from earlier in the conversation, 1 to 16; `0` off; `-1` (default) chooses: 8 with the whole model in VRAM, 6 for hybrid models, 4 for MoE models with experts in RAM |
| `device` | `auto` (default) or `gpu`: the best GPU, plus other GPUs of the same kind as further VRAM tiers; `cpu`; or one device by its `id` |
| `hot_experts` | `true`: with experts in RAM, put this model's most-used experts in VRAM, from what tierlm has learned about it. Default `false`: whether it is faster depends on the PC, and `/api/autotune` saves `true` only where it measured fastest |
| `live_experts` | `true`: as `hot_experts`, then while the model runs, experts the conversation keeps using move into VRAM in place of ones it stopped using (uses more RAM: the VRAM experts stay in RAM too). Works before anything is learned. Default `false`; `/api/autotune` measures it as a fourth split mode |
| `split_mode` | When the model doesn't fit in VRAM: `auto` (default: `experts` for MoE models, `layers` for the rest), `layers` (whole layers in VRAM, the first ones in RAM) or `experts` (attention and context in VRAM, experts in RAM) |
| `vram_limit_mb` | Use at most this much VRAM; 0 (default) uses what's free |

Any loaded model is freed first. The settings are saved for this model. Returns
`{"ok": true, "device": "...", "load_s": 3.2}`, or an `error` with HTTP 400/500.

### POST /api/autotune

Finds the fastest settings for a model on this PC. The body is the same as `/api/load`, plus the chat's sampling
(`temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`), which every measurement uses: each one answers two
questions exactly as `/api/chat` would. The reply is a stream of
server-sent events: `{"type":"status","text":"Measuring experts in RAM"}` while it works,
`{"type":"result","label":"GPU + RAM, experts in RAM, 6 CPU threads, no speculative decoding","tps":17.0}` for each measurement (the label lists every setting it was measured with), then
`{"type":"done","tps":22.3,"settings":{...},"results":[...]}` with the settings it saved for the model (`split_mode`,
`threads`, `speculative`, `draft_max`, `hot_experts`, `live_experts`), or `{"type":"error","message":"..."}`. The model stays loaded with the winner.

### POST /api/bench

Measures the loaded model with its current settings: reads a 512-token prompt, then answers two questions exactly
as `/api/chat` would. The optional body sets the sampling to measure with (`temperature`, `top_k`, `top_p`, `min_p`,
`repeat_penalty`, as in `/api/chat`). `gen_tps` is the speed a chat runs at.
Returns `{"prompt_tps": 143.0, "gen_tps": 22.3, "speculative": true, "draft_max": 2, "threads": 8}`.

### POST /api/spec

Turns speculative decoding on or off, or changes the guesses per check, without reloading:
`{"speculative": true, "draft_max": 2}`. Only when the model was loaded with its draft layer (`mtp_loaded` in
`/api/info`); otherwise it answers 409. `{"lookup": 8}` (or `0`, or `-1` for automatic) changes prompt lookup, for
any model, and keeps the conversation in the cache.

### POST /api/usage/reset

Forgets the loaded model's expert usage counts (and deletes their file). The next load places whole expert layers
again until it learns.

### GET /api/progress

`{"loading": true, "progress": 0.42}` while a model loads (the share of its weights read so far). Answers even while
the engine is busy.

### GET /api/log?after=N

The engine's recent log lines (what the black window shows), after line number `N` (-1 for all that are kept):
`{"lines": [{"n": 12, "text": "reply: 212 tokens at 22.3 tok/s (end)", "v": false}], "next": 13}`. `v` marks lines
from ggml and the graphics drivers (the console's Verbose view).

### GET /api/context, POST /api/context

Changes the context length while the model stays loaded: the cache is made again at the new size on the same devices
(VRAM and RAM as they are), keeping what's in it, in about a second.

- `GET /api/context?ctx=N` (tokens per connection): what it would take. `{"ctx", "current", "connections", "trained",
  "vram_change", "ram_change", "vram_free", "ram_free", "bytes_per_token", "live", "reload", "why"}`. `live`: it fits
  where the cache is now and can be done right away. `reload`: it needs more VRAM than is free, so loading the model
  again (which would move part of it to RAM) is the way; `why` says so, or why it isn't possible (longer than the
  model was trained for, or shorter than the conversations in the cache).
- `POST /api/context {"ctx": N}`: does it (409 with the reason if it can't), and saves N as this model's context for
  next time. `{"ok": true, "ctx": N, "seconds": ...}`. Waits for running replies to finish. On this computer only.

### Get models (`/api/hf`)

Searching Hugging Face for GGUF models and downloading them (the setup screen's **Get models…** window). The server
asks Hugging Face's public API through curl (Windows 10 and 11 include it). On this computer only.

- `GET /api/hf/search?q=...&sort=downloads|trendingScore|likes|lastModified`: `{"models": [{"id", "downloads", "likes", "updated"}]}`.
- `GET /api/hf/repo?id=owner/name`: its GGUF files grouped: `"quants"` (each `{"label", "size", "files": [{"path", "size", "sha256"}]}`,
  split files together, smallest first), `"vision"` (mmproj files), `"draft"` (mtp- files); and what the model is, from the
  first model file's header (its first megabytes): `"arch"`, `"supported"`, `"reason"`, `"kind"`, `"layers"`, `"experts"`,
  `"kv_bytes_per_token"`, `"ctx_train"`; `"gated"` when it needs a license accepted and a sign-in (not supported yet).
- `POST /api/hf/download {"id": "owner/name", "files": ["path", ...]}`: downloads them into tierlm's own `models`
  folder (next to it, whichever folder is chosen for browsing), in `owner\name\`. Sizes and checksums come from Hugging Face, not the request. Each file goes to a `.part` name that
  continues where it left off after an interruption, is checked against Hugging Face's SHA-256, then gets its name.
  One download at a time (409 while one runs); 507 when the drive hasn't room.
- `GET /api/hf/download`: `{"state": "idle|downloading|done|cancelled|error", "id", "error", "files": [{"path", "size", "done", "state"}], "done", "total", "folder"}`.
- `POST /api/hf/cancel`: stops it; what was downloaded is kept.

`TIERLM_HF_BASE` (tests) points all of this at another address in place of `https://huggingface.co`.

### GET /api/drives, POST /api/ssd

The SSD tier's drive (see [Getting started](GETTING_STARTED.md#models-bigger-than-your-ram-the-ssd-tier)).

- `GET /api/drives`: `{"drives": [{"path", "label", "free", "total"}], "ssd_dir", "ssd_free"}`: the drives the SSD tier
  can use (fixed drives on Windows) and the folder chosen (`""`: each model's own drive, read where the file is).
- `POST /api/ssd {"dir": "D:\\"}`: choose one. A drive's root gets a `tierlm-ssd` folder; `""` goes back to the
  model's own drive. Saved; used from the next load. On this computer only.

`/api/info` says what's there: `ssd_weights` (bytes), `ssd_layers`, `ssd_file`. `/api/load` takes `"ssd": false` to
keep everything in RAM for that load.

### POST /api/unload

Frees the model and everything that belongs to it (its vision file, draft block and the replies' caches), so its VRAM
and RAM are free again; replies still being written finish first. `{"embedding": true}` frees the embedding model too
(it stays the saved choice). The page's **Eject** button sends that. Returns `{"ok": true}`.

### POST /api/chat

Request:

```json
{
  "messages": [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 2048
}
```

- `messages` is the whole conversation; roles are `system`, `user` and `assistant`. The server formats it
  with the model's own chat format (ChatML, Llama 3, Tulu, `[INST]`). A message may carry `"images": ["data:image/png;base64,..."]`
  (models that read images): they go before its text. The `/v1` content parts work here too.
- `chat_id` (tierlm's page, on this computer): the saved chat this request continues. If it has an archive (below),
  its summary is added to the system message and the archived passages that best match the new message are put just
  before it (a `recall` event lists them).
- `overflow`: `"archive"` lets tierlm leave out the oldest messages when the conversation doesn't fit (down to
  about 60% of the context, so it isn't needed again at once); an `archived` event says how many, so the page can
  mark them archived. Without it, a conversation that doesn't fit gives an `error` event with `"code": "context_full"`.
- `temperature` 0 means greedy (always the most likely token). Defaults: 0.7, top_p 0.9, and no limit on `max_tokens` (0 or less, or leaving it out, means the reply runs until the model finishes or the context is full).
- Whatever part of the conversation is already in the KV cache is reused, so follow-up messages only process
  the new text.

Errors before streaming starts: `400` for a malformed body. If a reply is already being generated or a model is
loading, the request waits in line and the stream starts with `queue` events (below). No model loaded, or waiting
more than 30 minutes, gives an `error` event.

#### Response stream

The response is `text/event-stream`: lines of `data: {json}` separated by blank lines.

**0. `queue`**, only while waiting in line, each time the number ahead changes (`0`: just the reply running now):

```json
{"type": "queue", "ahead": 2}
```

**1. `prompt`**, once, after the conversation has been read:

```json
{"type": "prompt", "n_prompt": 58, "n_new": 31, "reused": 27, "prompt_s": 0.31, "prompt_tps": 100.0,
 "n_past": 58, "counts": [ ... n_layer * n_expert integers ... ]}
```

`counts[layer * n_expert + expert]` is how many times each expert was picked while reading the new prompt
tokens.

**2. `token`**, once per generated token:

```json
{"type": "token", "text": " Paris", "tps": 61.2, "n_past": 59,
 "routing": [[12, 40, 3, 57, 9, 61, 22, 0], ... one list of experts_used ids per layer ...]}
```

- `text` is the new text since the previous event. Multi-byte characters are never split, so `text` can be
  empty for a token that completes a character later.
- `routing[layer]` lists the experts the router picked for this token in that layer, best first.
- `tps` is the average generation speed of this reply so far.

**3. `done`**, at the end:

```json
{"type": "done", "n_gen": 42, "gen_s": 0.7, "gen_tps": 60.0, "stop": "end", "n_past": 100}
```

With speculative decoding or prompt lookup, `done` also has
`"spec": {"checks": 21, "drafted": 63, "accepted": 41, "looked": 30, "look_ok": 25}`: the model's checks, the tokens
guessed, and how many of them the model kept (each check also adds one token of its own, so tokens per check =
`n_gen / checks`); `looked` and `look_ok` are the part of those guesses that came from prompt lookup. Token events then carry no `routing`.

`stop` is `end` (the model finished), `length` (hit `max_tokens`), `context` (the context window is full)
or `abort` (the client disconnected).

If something goes wrong after streaming starts, the server sends `{"type": "error", "message": "..."}`.

#### Stopping a reply

Close the connection (for example, abort the `fetch`). The server notices on the next token and stops.

#### Example with curl

```bash
curl -N http://127.0.0.1:8040/api/chat -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello"}],"max_tokens":20}'
```

### Saved chats (`/api/chats`)

For tierlm's own page on the computer running tierlm (the same checks as the workspace below; from other devices the
page keeps chats in that browser). Each chat is a folder in `chats` next to the settings file, named
`<date> <title> (<id>)`, holding `chat.json` (what the page needs), `chat.md` (the conversation as readable
Markdown, with links to attached files and to the workspace files the model worked on) and `files/` (attached files).

- `GET /api/chats`: `{"folder": "C:\\tierlm\\chats", "chats": [{"id", "title", "created", "updated", "model", "messages", "archived", "folder"}]}`, newest first
- `GET /api/chats/<id>`: the chat: `{"id", "title", "created", "updated", "model", "summary", "autoArchive", "messages": [...]}`;
  a message can have `time` (ms), `archived`, `files` (`[{"name", "size"}]`), `shown`, `meta`, `tool_calls`, `recall`
- `POST /api/chats/<id>`: saves the chat (writes `chat.json` and `chat.md`); needs the `X-Tierlm-Workspace: 1` header
- `DELETE /api/chats/<id>`: deletes its folder
- `POST /api/chats/<id>/files` `{"name": "notes.txt", "text": "..."}`: keeps an attached file with the chat
- `GET /api/chats/<id>/files/<name>`: an attached file
- `POST /api/chats-folder` (`{}` or `{"id": ...}`): opens the folder in Explorer (Windows)

**The archive.** When a conversation no longer fits the context, the page can archive its oldest messages (see
`overflow` above): they stay in the chat but aren't sent any more. For each new message, tierlm finds the archived
passages that best match it, by meaning with the embedding model if one is loaded, else by keywords, and gives them
to the model with the message. The page can also have the model write a short summary of the archive, which is given
to the model from then on.

### The workspace (`/api/workspace`)

For tierlm's own chat page on the computer running tierlm. Every request must come from this computer (not through a
proxy or tunnel), be addressed to `127.0.0.1` / `localhost` / `[::1]` on tierlm's port, come from tierlm's page if the
browser sends an `Origin`, and carry the header `X-Tierlm-Workspace: 1`. Anything else gets 403.

- `GET /api/workspace`: `{"available": true, "folder": "C:\\code\\app", "exists": true, "picker": true}`.
- `POST /api/workspace` `{"folder": "C:\\code\\app"}` chooses the folder (saved); `""` clears it.
- `POST /api/workspace/pick` opens a folder picker on this computer (Windows) and returns `{"cancelled", "folder"}`.
- `POST /api/workspace/tool` `{"name": "read_file", "arguments": {"path": "src/app.py"}}` runs one of the model's
  tools: `list_files`, `read_file`, `search_files` return `{"result"}`. `write_file` and `edit_file` return
  `{"diff", "hash", "path", "created"}` and change nothing; sent again with `"apply": true, "hash": "<hash>"` they make
  the change (if the file still matches the hash) and add `"result"`. Problems come back as `{"error"}`.

`POST /api/chat` with `"workspace": true` gives the model these tools and tells it about the folder; its `done`
event then also has `raw` (the reply as the model wrote it, to send back as the assistant message), `text` (as
shown) and `tool_calls` (`[{"name", "arguments", "arguments_json"}]`). The page runs the calls through
`/api/workspace/tool`, adds `{"role": "tool", "name", "content"}` messages and asks again.

### The embedding model (`/api/embed`)

- `GET /api/embed`: the loaded embedding model: `{"loaded": true, "id", "name", "arch", "dims", "max_tokens", "device",
  "on_gpu", "bytes", "pooling"}`, or `{"loaded": false}`.
- `POST /api/embed/load` `{"model": "<id from /api/models>"}`: loads it (replacing the one loaded), on the GPU when
  there's room to spare after the chat model, else on the CPU. The choice is saved and loads again when tierlm starts.
- `POST /api/embed/unload`.

In `/api/models`, embedding models have `"kind": "embedding"`, `"embedding": true` when tierlm can load them, and
`"dims"`. `/v1/models` lists the loaded embedding model with `"type": "embeddings"` (the chat model has `"type": "llm"`).

### POST /api/embed/download, GET /api/embed/download

`POST` (this computer only) downloads nomic-embed-text-v1.5 Q8_0 from `huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF`
into the models folder, with curl (part of Windows 10 and 11). The file is checked against the size and SHA-256
Hugging Face gives for it (`X-Linked-Size`, `X-Linked-Etag`) before it gets its name; a file that doesn't match is
deleted. `GET` reports `{"state": "idle" | "downloading" | "checking" | "done" | "error", "bytes", "total", "error",
"file"}`; then `POST /api/embed/load {"model": file}` loads it (the page does that by itself).

### POST /api/reset

Clears the KV cache. Returns `{"ok": true}`.

### GET /api/network

```json
{"allow_network": false, "listening_on": "127.0.0.1", "port": 8040, "lan_address": "192.168.1.20",
 "api_key_set": true, "api_key": "tlm-...", "fixed_by_command_line": false,
 "urls": [{"label": "This computer", "url": "http://127.0.0.1:8040/v1"},
          {"label": "Other devices on your network", "url": "http://192.168.1.20:8040/v1"}]}
```

`api_key` itself is only included for requests from this computer.

### POST /api/network

`{"allow_network": true}` and/or `{"api_key": "generate"}` (a new random key), `{"api_key": "my-key"}` or
`{"api_key": ""}` (no key), and/or `{"api_trim_oldest": true}`: when an app's conversation is too long for the
context, leave out its oldest messages instead of returning an error (off by default: coding agents shorten their
conversations themselves when told). Only accepted from this computer. Returns the same as `GET /api/network`.
