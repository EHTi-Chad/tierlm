# tierlm

**A local LLM inference engine for mixture-of-experts models, built to spread big models across VRAM, RAM and SSD.**

tierlm is written from scratch in C++ by **Claude** (Anthropic's AI), with **Chad Jameson (EHTi)** directing
the project, making the design calls, and testing and debugging it on real hardware. It uses
[ggml](https://github.com/ggml-org/ggml)'s math kernels, but the tokenizer, model graph, KV cache, attention,
memory planner, server and dashboard are all tierlm's own. It does not use llama.cpp.

It runs the common model families (**Llama 2/3, Mistral, Mixtral, Qwen2/2.5, Qwen3, Qwen3 MoE, Qwen3.5/3.6/3.8
dense and MoE, OLMoE**) on the
CPU, on any GPU with Vulkan (Intel, AMD, NVIDIA), or on NVIDIA with CUDA. **Models bigger than the GPU's memory
are split between VRAM and RAM automatically**: on a laptop RTX 3070 (8 GB), the 14.7 GB Qwen3-30B-A3B runs at
about **27 tokens/second**, 40% faster than on the CPU alone. A web page lets you pick a model and chat, with a
live dashboard showing speed, where the model lives, and the router picking experts.

**New user? Read the [tierlm 0.2.0 guide (PDF)](docs/tierlm-0.2.0-guide.pdf)**: what tierlm is, step-by-step
setup, every setting, how it works, and all test results, with diagrams.

> **Status: early.** VRAM and RAM tiering works; the SSD tier and a hot-expert cache are next. Not every model
> family is supported yet. See [Supported models](#supported-models) and the [roadmap](#roadmap).

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
- **Own tokenizers**: byte-level BPE (GPT-2, Llama 3 and Qwen splitting) and SentencePiece, matching Hugging
  Face's tokenizers exactly on real Qwen3, Llama 3.2 and TinyLlama models.
- **Own model graph for every supported family**: dense and mixture-of-experts, grouped-query attention, QK-norm,
  RoPE variants, verified against independent NumPy implementations, including when split between GPU and RAM.
- **Split (multi-part) GGUFs**: models published as `-00001-of-0000N.gguf` files load as one model.
- **Tier loader**: a model that doesn't fit in VRAM is split by whole layers (or with experts in RAM), with
  each RAM layer pinned to the CPU so a token changes device only once.
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

**RTX 3070 Laptop (8 GB) + Ryzen 9 5900HX, Windows 11:**

| Model | Where it runs | Generation |
|---|---|---|
| Llama 3.2 1B Q4_K_M | VRAM, Vulkan | 230.7 tok/s |
| Llama 3.2 1B Q4_K_M | VRAM, CUDA | 198.1 tok/s |
| Qwen3-30B-A3B Q3_K_M (14.7 GB) | CPU only | 19.3 tok/s |
| Qwen3-30B-A3B Q3_K_M (14.7 GB) | 22 of 48 layers in VRAM, rest in RAM (CUDA) | **27.1 tok/s** |

**Intel Arc 140V (integrated), Intel Core Ultra 7 258V:**

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

**tierlm's point is running models bigger than your GPU.** It fills GPU memory first, then puts the rest of the
model in system RAM, automatically. On a laptop RTX 3070 with 8 GB, the 14.7 GB Qwen3-30B-A3B runs at about
27 tokens/second.

**What you need**

- **Windows 10 or 11, 64-bit**, with an AVX2 processor (almost any Intel or AMD CPU from 2015 on).
- **System RAM: at least 32 GB** to run models bigger than your GPU; **64 GB or more is recommended**. The
  model has to fit in GPU memory and RAM together, with about 8 GB left over for Windows and your other apps.
- **A GPU** (optional, but much faster): an NVIDIA GeForce RTX card, Intel Arc or AMD Radeon. Without one,
  tierlm runs on the CPU.

| GPU memory | RAM | Biggest model, roughly |
|---|---|---|
| 8 GB | 32 GB | about 30 GB |
| 8 GB | 64 GB | about 60 GB |
| 16 GB | 64 GB | about 70 GB |
| 32 GB | 64 GB | about 85 GB |

Mixture-of-experts models (names with **A3B**, **A10B** and so on, such as Qwen3-30B-A3B or Qwen3.6-35B-A3B)
only use a small part of the model for each word, so they stay fast even when most of the model is in RAM.
These are the best fit for tierlm. Ordinary (dense) models also work, but slow down a lot once most of the
model is in RAM.

**Steps**

1. From [Releases](https://github.com/EHTi-Chad/tierlm/releases), download the zip for your graphics card and
   extract it:
   - **`tierlm-<version>-windows-nvidia.zip`**: NVIDIA GeForce RTX 20, 30, 40 and 50 series. Uses CUDA, the
     fastest option on NVIDIA. Everything it needs is included; you only need a current NVIDIA driver.
   - **`tierlm-<version>-windows-intel-amd.zip`**: Intel Arc, AMD Radeon, older NVIDIA cards (GTX 10 series
     and earlier), or no graphics card.
2. Get a model. tierlm uses **GGUF** files, the same ones LM Studio uses; if you already have LM Studio models,
   tierlm can use that folder directly. To download one, pick a size for your computer:
   - Bigger than your GPU, the real test: [`Qwen3-30B-A3B` Q3_K_M](https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF/tree/main)
     (14.7 GB; needs 32 GB RAM with an 8 GB GPU)
   - The newest: [`Qwen3.6-35B-A3B`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main) Q4_K_M (about 21 GB) or
     Q8_0 (about 35 GB, for 64 GB RAM). Support is brand new; the first real-model test is under way.
   - Just to check it works: [`Llama-3.2-1B-Instruct` Q4_K_M](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/tree/main) (0.8 GB)

   Put the file in tierlm's `models` folder, or leave it where it is and choose that folder in step 3.
3. Double-click `start-tierlm.bat`. Your browser opens at http://127.0.0.1:8040, where tierlm shows the
   hardware it found and the models in your folder (the **⋯** menu changes the folder). Pick a model and chat.

Step-by-step guide: **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** or the [PDF guide](docs/tierlm-0.2.0-guide.pdf) · Problems: [docs/FAQ.md](docs/FAQ.md)

### Supported models

| Family | Tested on real weights |
|---|---|
| OLMoE | OLMoE-1B-7B-0924-Instruct |
| Qwen3, Qwen3 MoE | Qwen3-0.6B, Qwen3-30B-A3B |
| Llama 3.x | Llama-3.2-1B-Instruct |
| Llama 2 / Mistral 7B / TinyLlama (SentencePiece) | TinyLlama-1.1B-Chat |
| Mixtral, Qwen2/2.5 | model math verified, real weights not yet tested |
| Qwen3.5 / 3.6 / 3.8 dense (`qwen35`, hybrid Gated DeltaNet) | model math verified, real weights not yet tested |
| Qwen3.5 / 3.6 MoE (`qwen35moe`, e.g. Qwen3.6-35B-A3B) | model math verified, real weights not yet tested |

Not yet: Qwen3-Next / Qwen3-Coder-Next (`qwen3next`), Gemma, Phi, DeepSeek, gpt-oss and others. The setup screen says which
of your files are supported. Full list: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md#supported-models).

## Build from source (Windows)

Full instructions, including Linux/WSL and common pitfalls, are in [docs/BUILD.md](docs/BUILD.md).

```bat
git clone --recurse-submodules https://github.com/EHTi-Chad/tierlm.git C:\dev\tierlm
cd /d C:\dev\tierlm
cmake -S . -B build -G Ninja -DTIERLM_VULKAN=ON -DTIERLM_CUDA=ON
cmake --build build
build\tierlm-devices.exe
build\tierlm-server.exe --models C:\models --port 8040
```

Then open http://127.0.0.1:8040. Run these from a Visual Studio Developer Command Prompt with the Vulkan SDK
installed (and the CUDA Toolkit for `-DTIERLM_CUDA=ON`; leave it out for a Vulkan-only build).

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

- **[docs/tierlm-0.2.0-guide.pdf](docs/tierlm-0.2.0-guide.pdf)**: the full guide with diagrams and test results
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
- [x] Qwen3.5 / 3.6 / 3.8 dense: hybrid Gated DeltaNet + gated attention, verified against a NumPy reference
- [x] Qwen3.5 / 3.6 MoE: the same hybrid with routed experts and a gated shared expert
- [ ] Real-model testing of the new families; more families (Qwen3-Next, Gemma, Phi, DeepSeek, gpt-oss)
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
