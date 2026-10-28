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
| POST | `/api/chat` | Generate a reply, streamed as Server-Sent Events |
| POST | `/api/reset` | Clear the KV cache |
| GET | `/api/network` | Network access, API key status, and the addresses apps should use |
| POST | `/api/network` | Change network access or the API key (this computer only) |
| GET | `/v1/models` | OpenAI-compatible: models |
| POST | `/v1/chat/completions` | OpenAI-compatible: chat, streamed or not |

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

The loaded model first (`"loaded": true`), then every supported model in the models folder. `id` is the
model's path inside the folder, e.g. `lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf`.

### POST /v1/chat/completions

| Field | Notes |
|---|---|
| `model` | An `id` from `/v1/models` loads that model first (with the settings last used for it in tierlm), which takes as long as loading it on the setup screen. Any other name, such as `gpt-4o`, uses the loaded model. |
| `messages` | `system`, `developer` (same as system), `user`, `assistant`. Content can be a string or a list of text parts; images are skipped (no vision yet). `tool` messages are passed to the model as user text. |
| `stream` | `true` streams `chat.completion.chunk` events, ending with `data: [DONE]`. `stream_options: {"include_usage": true}` adds a final chunk with token counts. |
| `max_tokens`, `max_completion_tokens` | Longest reply. Left out: no limit (until the model finishes or the context is full). |
| `temperature`, `top_p`, `seed`, `stop` | As in OpenAI. `stop` is a string or a list of strings; the reply ends just before one. |
| `top_k`, `min_p`, `repeat_penalty` | Extras, as in LM Studio and llama.cpp. `repeat_penalty` is off (1.0) unless given, because it harms code. |

The reply includes `usage` (`prompt_tokens`, `completion_tokens`). `finish_reason` is `stop` (the model
finished, or a stop string), or `length` (`max_tokens`, or the context window is full).

One reply is generated at a time. Requests from apps wait their turn; while one is running, the web page's
own chat answers "busy". Requests share the model's memory of the conversation, so an app that resends a
growing conversation only pays for the new part.

Not yet: tool (function) calling, which coding agents such as Cursor's agent mode, Cline and Continue's agent
mode need; `logprobs`; `n` > 1; embeddings; images.

Errors use OpenAI's format: `{"error": {"message": "...", "type": "...", "code": ...}}`.

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
them. Network settings can only be changed on the computer running tierlm.

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
  "version": "0.2.0"
}
```

While a reply is streaming or a model is loading, `/api/info` returns the last known details with
`"busy": true` instead of waiting.

### GET /api/devices

```json
{"devices": [
   {"id": "Vulkan0", "description": "Intel(R) Arc(TM) 140V GPU (16GB)", "type": "igpu", "backend": "Vulkan",
    "free": 17179869184, "total": 19219415040},
   {"id": "CPU", "description": "Intel(R) Core(TM) Ultra 7 258V", "type": "cpu", "backend": "CPU",
    "free": 11811160064, "total": 33822867456}],
 "threads": 4, "hw_threads": 8, "version": "0.2.0", "warnings": []}
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
| `kv_type` | `f16` or `f32` KV cache |
| `batch` | Tokens per step when reading a prompt (default 256) |
| `experts_used` | Active experts per token for MoE models; 0 keeps the model's own value |
| `seed` | -1 for random sampling; any other number makes replies repeatable |

Any loaded model is freed first. The settings are saved for this model. Returns
`{"ok": true, "device": "...", "load_s": 3.2}`, or an `error` with HTTP 400/500.

### POST /api/unload

Frees the model and its memory. Returns `{"ok": true}`.

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
  with the model's own chat format (ChatML, Llama 3, Tulu, `[INST]`).
- `temperature` 0 means greedy (always the most likely token). Defaults: 0.7, top_p 0.9, and no limit on `max_tokens` (0 or less, or leaving it out, means the reply runs until the model finishes or the context is full).
- Whatever part of the conversation is already in the KV cache is reused, so follow-up messages only process
  the new text.

Errors before streaming starts: `400` for a malformed body, `409` if a reply is already being generated, a
model is loading, or no model is loaded.

#### Response stream

The response is `text/event-stream`: lines of `data: {json}` separated by blank lines.

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
`{"api_key": ""}` (no key). Only accepted from this computer. Returns the same as `GET /api/network`.
