# Server API

`tierlm-server` serves the web page and a small JSON API on the same port (default
`http://127.0.0.1:8080`). There is no authentication; keep it on localhost or a trusted network.

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

## GET /api/info

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

## GET /api/devices

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

## GET /api/models

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

`id` is the model's path relative to the folder; pass it to `/api/load`. With `recursive` on, subfolders are
searched too (up to 4 levels, as LM Studio stores models as `publisher/model/file.gguf`). `suggestions` lists
other model folders found on this computer. `saved` holds the settings last used for each model, by `id`.

## POST /api/folder

```json
{"folder": "D:\\AI\\models", "recursive": true}
```

Either field can be left out. The choice is saved in `tierlm-settings.json` (in the folder tierlm runs from)
and used next time, unless `--models` is given on the command line.

## POST /api/pick-folder

Opens the Windows folder picker on the computer running tierlm and returns
`{"cancelled": false, "folder": "D:\\AI\\models"}`; it doesn't change the folder by itself. Only answers
requests from the same computer. Elsewhere it returns HTTP 501.

## POST /api/load

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

## POST /api/unload

Frees the model and its memory. Returns `{"ok": true}`.

## POST /api/chat

Request:

```json
{
  "messages": [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 512
}
```

- `messages` is the whole conversation; roles are `system`, `user` and `assistant`. The server formats it
  with the OLMoE-Instruct (Tulu) chat template.
- `temperature` 0 means greedy (always the most likely token). Defaults: 0.7, top_p 0.9, max_tokens 512.
- Whatever part of the conversation is already in the KV cache is reused, so follow-up messages only process
  the new text.

Errors before streaming starts: `400` for a malformed body, `409` if a reply is already being generated, a
model is loading, or no model is loaded.

### Response stream

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

### Stopping a reply

Close the connection (for example, abort the `fetch`). The server notices on the next token and stops.

### Example with curl

```bash
curl -N http://127.0.0.1:8040/api/chat -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello"}],"max_tokens":20}'
```

## POST /api/reset

Clears the KV cache. Returns `{"ok": true}`.
