# Server API

`tierlm-server` serves the web page and a small JSON API on the same port (default
`http://127.0.0.1:8080`). There is no authentication; keep it on localhost or a trusted network.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | The web page (`web/index.html`) |
| GET | `/api/info` | Model, device and memory details |
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

Errors before streaming starts: `400` for a malformed body, `409` if a reply is already being generated.

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
