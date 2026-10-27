# Server API

`tierlm-server` serves the web page and a small JSON API on the same port (default
`http://127.0.0.1:8080`). There is no authentication; keep it on localhost or a trusted network.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | The web page (`web/index.html`) |
| GET | `/api/info` | Model, device and memory details (`{"loaded": false}` before a model is loaded) |
| GET | `/api/devices` | GPUs and CPU the engine can use, with free memory |
| GET | `/api/models` | `.gguf` files in the models folder, and whether each is supported |
| POST | `/api/load` | Load a model on a device |
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
  "kv_type": "f16", "flash_attn": true
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
 "threads": 4, "hw_threads": 8}
```

`type` is `gpu` (dedicated), `igpu` (integrated, shares system memory) or `cpu`. Pass `id` as the `device`
when loading.

## GET /api/models

```json
{"folder": "C:\\tierlm\\models", "default_ctx": 4096, "current": "",
 "models": [
   {"file": "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", "name": "OLMoE 1B 7B 0924 Instruct", "arch": "olmoe",
    "supported": true, "bytes": 4212186208, "layers": 16, "experts": 64, "experts_used": 8,
    "ctx_train": 4096, "kv_bytes_per_token": 131072},
   {"file": "Llama-3.2-1B-Q4_K_M.gguf", "arch": "llama", "supported": false,
    "reason": "the llama architecture is not supported yet", "bytes": 807694464}]}
```

Only the file headers are read, so this is fast. `kv_bytes_per_token` is the KV cache cost per token of
context (f16), useful for estimating memory: model bytes + `kv_bytes_per_token` × context.

## POST /api/load

```json
{"model": "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", "device": "Vulkan0", "ctx": 4096, "threads": 4}
```

`model` is a file name inside the models folder (other paths are refused). `device` is an `id` from
`/api/devices`, or `auto` / `cpu` / `gpu`. Any loaded model is freed first. Returns
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
