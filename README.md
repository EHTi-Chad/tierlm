# tierlm

**A local LLM inference engine for mixture-of-experts models, built to spread big models across VRAM, RAM and SSD.**

tierlm is written from scratch in C++ by **Claude** (Anthropic's AI), with **Chad Jameson (EHTi)** directing
the project, making the design calls, and testing and debugging it on real hardware. It uses
[ggml](https://github.com/ggml-org/ggml)'s math kernels, but the tokenizer, model graph, KV cache, attention,
memory planner, server and dashboard are all tierlm's own. It does not use llama.cpp.

It currently runs **OLMoE-1B-7B** (a 64-expert MoE model) on the CPU or on any GPU with Vulkan, including the
Intel Arc 140V integrated GPU, where it generates at **about 68 tokens/second**. A built-in web page lets you
chat with the model while a live dashboard shows the router picking experts for every token.

> **Status: early.** One model family is supported, and VRAM/RAM/SSD tiering is planned and modelled but not
> yet loading weights across tiers. See the [roadmap](#roadmap).

## Why

Mixture-of-experts models split each layer into many small "expert" networks and use only a few per token:
OLMoE uses 8 of 64. Experts are over 90% of the file, and most are idle at any moment. So if an engine keeps
the busiest experts in the fastest memory and the rest further away, a model much bigger than the GPU can
still run quickly. That is the idea behind [Strata](https://github.com/Niko1221/Strata), and tierlm is an
independent engine built on the same principle.

## Features

- **Own GGUF reader and model map**: lists every tensor's size, type, position, layer and role.
- **Placement planner** (dry run): splits experts into individual units and assigns each to VRAM, RAM or SSD
  within a memory budget, and estimates per-token traffic from each tier.
- **Own byte-level BPE tokenizer**: matches Hugging Face's tokenizer exactly on all test cases, including
  added tokens, accents, CJK, emoji and code.
- **Own OLMoE graph**: QK-norm attention, RoPE, top-8 of 64 expert routing, verified against an independent
  NumPy implementation.
- **KV cache in f16** and **flash attention**, with prompts processed in batches.
- **Runs on any ggml device**: CPU, Vulkan (Intel, AMD, NVIDIA) and CUDA builds, with automatic CPU fallback for
  any operation a device can't run.
- **Setup screen**: detects every GPU (Intel, AMD, NVIDIA) and the CPU with free memory and places the model
  automatically (GPU first, then CPU), lists the GGUF models in any folder you point it at (including LM
  Studio's), and remembers LM Studio-style load settings per model.
- **Web interface**: streaming chat, plus a live dashboard with an expert activity map, speed vs. context,
  memory placement, and a "hot-expert cache" preview built from real routing data.
- **Local HTTP API** with streaming replies (Server-Sent Events): see [docs/API.md](docs/API.md). OpenAI-compatible endpoints are on the roadmap.

## Benchmarks

OLMoE-1B-7B-0924-Instruct, Q4_K_M (3.9 GiB), generation speed as context grows (`tierlm-bench`):

| Context | Intel Arc 140V, before | Intel Arc 140V, now | Gain |
|---|---|---|---|
| 64 tokens | 45.3 tok/s | **67.8 tok/s** | 1.5× |
| 512 | 35.4 | **43.0** | 1.2× |
| 1,024 | 11.8 | **33.7** | 2.9× |
| 2,048 | 12.9 | **22.4** | 1.7× |

"Before" is the first version (f32 KV cache, plain attention); "now" adds the f16 KV cache, flash attention
and reserved compute memory. Laptop: Intel Core Ultra 7 258V, 32 GB shared memory, Windows 11, Vulkan driver
101.8826. On the same laptop's CPU alone, tierlm generates about 40–46 tok/s at short context.

Long contexts get slower because every new token has to read the whole KV cache. OLMoE keeps separate keys and
values for all 16 attention heads (about 128 KB per token of context at f16), so a quantized KV cache is the
next lever for it.

## Try it (Windows)

1. Download the latest `tierlm-<version>-windows-x64-vulkan.zip` from
   [Releases](https://github.com/EHTi-Chad/tierlm/releases) and extract it.
2. Download [`OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf`](https://huggingface.co/bartowski/OLMoE-1B-7B-0924-Instruct-GGUF/tree/main)
   (4.2 GB) into its `models` folder.
3. Double-click `start-tierlm.bat`. Your browser opens at http://127.0.0.1:8040, where you pick your GPU (or
   CPU) and the model, then chat.

Step-by-step guide: **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** · Problems: [docs/FAQ.md](docs/FAQ.md)

### Supported models

Only the **OLMoE** family for now: `OLMoE-1B-7B-0924-Instruct` (Q4_K_M tested; other quantizations should
work). Other GGUF models are not supported yet and fail with "expected olmoe". Qwen3 MoE is next.

## Build from source (Windows, Vulkan)

Full instructions, including Linux/WSL and common pitfalls, are in [docs/BUILD.md](docs/BUILD.md).

```bat
git clone --recurse-submodules https://github.com/EHTi-Chad/tierlm.git C:\dev\tierlm
cd /d C:\dev\tierlm
cmake -S . -B build -G Ninja -DTIERLM_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
build\tierlm-devices.exe
build\tierlm-server.exe C:\models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf --port 8040
```

Then open http://127.0.0.1:8040. Run these from a Visual Studio Developer Command Prompt with the Vulkan SDK
installed.

## Tools

| Tool | What it does |
|---|---|
| `tierlm-server` | Web chat and live dashboard, plus a local HTTP API |
| `tierlm-run` | Generate text from a prompt on the command line |
| `tierlm-bench` | Measure generation speed at several context lengths |
| `tierlm-devices` | List the compute devices this build can use, and why a GPU is missing |
| `tierlm-inspect` | Summarise a GGUF model: size by role, experts, per-layer size |
| `tierlm-plan` | Dry-run the VRAM/RAM/SSD placement plan for a memory budget |
| `tierlm-kv` | Dump a GGUF file's metadata |
| `tierlm-tok-test`, `tierlm-logits` | Test harnesses used by `tests/` |

Details and options: [docs/TOOLS.md](docs/TOOLS.md).

## Documentation

- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md): download, run, and use tierlm (no programming needed)
- [docs/FAQ.md](docs/FAQ.md): questions and troubleshooting
- [docs/BUILD.md](docs/BUILD.md): building on Windows and Linux, GPU backends, making a release
- [docs/TOOLS.md](docs/TOOLS.md): every tool and option, environment variables, tests
- [docs/API.md](docs/API.md): the server's HTTP API and event stream
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how tierlm works, and design decisions

## Roadmap

- [x] GGUF model map and placement planner (dry run)
- [x] Byte-level BPE tokenizer, exact match with Hugging Face
- [x] OLMoE forward pass, KV cache, flash attention, f16 cache
- [x] Device-independent backend (CPU, Vulkan, CUDA builds)
- [x] Web chat + live dashboard
- [x] Setup screen: automatic hardware placement, models folder, per-model settings
- [x] Architecture layer: Llama 2/3, Mistral, Mixtral, Qwen2, Qwen3, Qwen3 MoE, OLMoE; Llama 3 / Qwen / SentencePiece tokenizers
- [ ] Real-model testing of the new families; more families (Qwen3.5 hybrid, Gemma, Phi, DeepSeek, gpt-oss)
- [x] Tier loader, first version: models bigger than VRAM split between VRAM and RAM (whole layers or experts), VRAM limit setting
- [ ] SSD tier (stream rarely used weights from disk) and a hot-expert cache in VRAM
- [ ] Search and download GGUF models from Hugging Face inside tierlm
- [ ] Routing statistics and a hot-expert cache that promotes busy experts to VRAM
- [ ] NVIDIA (CUDA) testing on an RTX 5090; AMD testing via Vulkan/ROCm
- [ ] Quantized KV cache, speculative decoding, OpenAI-compatible endpoints

## Credits and licence

- Code: written by Claude (Anthropic), directed and tested by Chad Jameson (EHTi).
- tierlm is released under the [MIT License](LICENSE). Copyright (c) 2026 Chad Jameson (EHTi).
- Third-party components (ggml, cpp-httplib, nlohmann/json) keep their own MIT licences; see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Model weights are not included and have their own licences.
