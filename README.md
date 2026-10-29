# tierlm

**A performance-first local LLM engine: it puts every part of a model in the preferred memory location. Big 
models run as fast as your PC allows, and your RAM, not your GPU, sets how big you can go.**

tierlm is written from scratch in C++ by **Claude** (Anthropic's AI), with **Chad Jameson (EHTi)** directing
the project, making the design calls, and testing and debugging it on real hardware. It uses
[ggml](https://github.com/ggml-org/ggml)'s math kernels, with a few small changes that are all listed and explained
in [docs/GGML_CHANGES.md](docs/GGML_CHANGES.md); the tokenizer, model graph, KV cache, attention, memory planner,
server and dashboard are all tierlm's own. It does not use llama.cpp.

It runs the common model families (**Llama 2/3, Mistral, Mixtral, Qwen2/2.5, Qwen3, Qwen3 MoE, Qwen3.5/3.6/3.8
dense and MoE, Qwen3-Next, OLMoE**) on the CPU, on any GPU with Vulkan (Intel, AMD, NVIDIA), or on NVIDIA with
CUDA, on **Windows** (ready-to-run download) and **Linux** (build from source; a ready-made download is coming). **Models bigger than the GPU's memory are split between VRAM and RAM automatically**: on a laptop RTX 3070
(8 GB), the 17 GB Qwen3.6-35B-A3B runs at about **33 tokens/second** (35 with speculative decoding), and on an
RTX 5090 the 46 GB Qwen3.6-35B-A3B Q8_0 ran at **36.6 tokens/second** with 128K of context (measured with 0.4.0;
0.5.0 is faster on NVIDIA, new numbers coming). A web page lets you pick a model and chat, with a
live dashboard showing speed, where the model lives, and the router picking experts.

**New user? Read the [tierlm 0.5.0 guide (PDF)](docs/tierlm-0.5.0-guide.pdf)**: what tierlm is, step-by-step
setup, every setting, how it works, and all test results, with diagrams.

> **Status: early.** VRAM and RAM tiering works, down to individual experts; the SSD tier is next. Not every model
> family is supported yet. See [Supported models](#supported-models) and the [roadmap](#roadmap).

## Why

Local AI apps are usually built around the graphics card: a model that fits in VRAM runs well, and anything bigger
is cut wherever VRAM runs out. But not every part of a model is equal. Attention, the router and the context are
used for every word; in a mixture-of-experts model, most experts sit idle most of the time. VRAM is fast but small,
RAM is big but several times slower, and an SSD is bigger again. **Speed comes from putting the busiest parts in the
fastest memory**, and that is what tierlm is built around (the tiering idea was inspired by
[Strata](https://github.com/Niko1221/Strata)):

| Memory | What tierlm puts there | Why |
|---|---|---|
| **VRAM** (GPU) | Attention, the router and the context (KV cache) of every layer, then as many experts or layers as fit | Used for every word, and the GPU reads its memory fastest |
| **System RAM** (CPU) | The rest: experts of an MoE model, or whole layers of a dense one | Needed less often; the CPU works on them with the best instructions it has (AVX2, AVX-512) |
| **SSD** (soon) | Rarely used experts, streamed on demand | For models bigger than VRAM and RAM together |

It makes a large difference: the 17 GB Qwen3.6-35B-A3B on an 8 GB laptop GPU runs at 13 tok/s split by whole layers,
33 tok/s with attention and context in VRAM and only the experts in RAM, and 35 tok/s with speculative decoding.

How tierlm keeps it fast, at any size:

- **Size is about memory.** Any model that fits in VRAM plus RAM runs (about 10 GB is needed for Windows, the
  context and working memory). The SSD tier, coming soon, will stream rarely used experts and push past RAM.
- **Each part goes where it's fastest.** For mixture-of-experts models, attention and the context stay on the GPU and
  only the experts go to RAM, so the CPU reads just the few experts each token picks; dense models are split by whole
  layers. tierlm picks the split itself.
- **More than one token per read.** When part of a model is in RAM, speed is set by how fast RAM can be read.
  Speculative decoding gets several tokens out of each read, and the CPU work uses the best instruction set the
  processor has (AVX2, AVX-512).

| Graphics card | System RAM | Largest model, roughly | For example |
|---|---|---|---|
| 8 GB | 32 GB | 30 GB | Qwen3.6-35B-A3B at 3 or 4 bits |
| 8 GB | 64 GB | 60 GB | Qwen3.6-35B-A3B at 8 bits |
| 16 GB | 64 GB | 70 GB | Qwen3-Coder-Next 80B-A3B at 4 to 5 bits |
| 24 GB | 64 GB | 75 GB | Llama 3.3 70B at 6 bits |
| 32 GB | 64 GB | 85 GB | Qwen3-Coder-Next 80B-A3B at 6 bits |
| 24 GB | 128 GB | 140 GB | Qwen3.5-122B-A10B at 8 bits |

Mixture-of-experts models (A3B, A10B...) stay fast with most of the model in RAM: Qwen3.6-35B-A3B (17 GB) runs at
33 tok/s on an 8 GB laptop RTX 3070. Dense models are slower once they spill into RAM, which is where speculative
decoding helps most.

## Features

- **Own GGUF reader and model map**: lists every tensor's size, type, position, layer and role.
- **Placement planner** (dry run): splits experts into individual units and assigns each to VRAM, RAM or SSD
  within a memory budget, and estimates per-token traffic from each tier.
- **Own tokenizers**: byte-level BPE (GPT-2, Llama 3 and Qwen splitting) and SentencePiece, matching Hugging
  Face's tokenizers exactly on real Qwen3, Llama 3.2 and TinyLlama models.
- **Own model graph for every supported family**: dense and mixture-of-experts, grouped-query attention, QK-norm,
  RoPE variants, verified against independent NumPy implementations, including when split between GPU and RAM.
- **Split (multi-part) GGUFs**: models published as `-00001-of-0000N.gguf` files load as one model.
- **Tier loader**: a model that doesn't fit in VRAM is split automatically: mixture-of-experts models keep attention
  and context in VRAM with the experts in RAM; dense models are split by whole layers,
  with each RAM layer pinned to the CPU so a token changes device only once.
- **KV cache in f16, 8-bit or 4-bit** and **flash attention**, with prompts processed in batches.
- **Every x86 CPU generation**: release builds carry a CPU engine per generation (plain x86-64 up to AVX2, AVX-512 and
  AMX) and load the best one at start-up; common quantizations are rearranged for the CPU's wide instructions on load
  (prompts read about 3.5× faster on the CPU).
- **Runs on any ggml device**: CPU, Vulkan (Intel, AMD, NVIDIA) and CUDA builds, with automatic CPU fallback for
  any operation a device can't run.
- **Setup screen**: detects every GPU (Intel, AMD, NVIDIA) and the CPU with free memory and places the model
  automatically (GPU first, then CPU), lists the GGUF models in any folder you point it at (including LM
  Studio's), and remembers LM Studio-style load settings per model.
- **CUDA graphs**: on NVIDIA cards each token's GPU work is recorded once and replayed with a single launch instead
  of thousands, which made generation about 50% faster than 0.4.0 on a laptop RTX 3070 with part of the model in RAM
  (21.7 → 33.3 tok/s, same results).
- **Hot experts (adaptive tiering)**: tierlm learns which experts each mixture-of-experts model actually uses as you
  chat, and can keep those individual experts in VRAM instead of whole expert layers, so most of the router's picks
  are served from fast memory. Whether that beats the plain split depends on the PC, so it is off by default and
  **Find the fastest settings** turns it on where it measures faster.
- **Several GPUs**: with more than one card of the same kind (two NVIDIA cards with CUDA, say), the model fills the
  first GPU, then the next, then RAM.
- **Find the fastest settings**: one click loads the model and measures the settings that affect speed on this PC
  (split mode, CPU threads, speculative decoding and how far ahead to guess), then keeps the fastest.
- **Web interface**: streaming chat with a collapsible side menu (live settings and system prompt, API, a console
  showing what the engine is doing, model info, saved chats and system-prompt presets, a one-click benchmark), a
  thin live view (speed, context, memory and an explained expert map that shows which experts work and where they
  live), and a light/dark/automatic theme.
- **Speculative decoding** with the model's own built-in draft layer (MTP, in Qwen3.5 / 3.6 / 3.8 and Qwen3-Next
  files, or a separate `mtp-*.gguf` next to the model): the model checks several guessed tokens in one pass, so
  each read of its weights from RAM produces more than one token. Same answers, faster replies; most useful when the
  model doesn't fit in VRAM. Plus **8-bit and 4-bit context memory** (KV cache) to fit more of the model in VRAM.
- **OpenAI-compatible API** (`/v1/models`, `/v1/chat/completions`, streamed or not): Open WebUI, Continue, LibreChat
  and anything built on the OpenAI SDK can use tierlm. An **Allow other devices** switch and an optional API key
  control who can connect. See [docs/API.md](docs/API.md) and [Connect other apps](docs/GETTING_STARTED.md#connect-other-apps).

## Benchmarks

**RTX 5090 (32 GB) + Ryzen 9 9950X3D, 64 GB DDR5, Windows 11:**

| Model | Where it runs | Generation |
|---|---|---|
| Qwen3.6-35B-A3B Q8_0 (46 GB file), 128K context | split between VRAM and RAM (CUDA) | **36.6 tok/s** |

**RTX 3070 Laptop (8 GB) + Ryzen 9 5900HX, Windows 11:**

| Model | Where it runs | Generation |
|---|---|---|
| Llama 3.2 1B Q4_K_M | VRAM, Vulkan | 230.7 tok/s |
| Llama 3.2 1B Q4_K_M | VRAM, CUDA | 198.1 tok/s |
| Qwen3-30B-A3B Q3_K_M (14.7 GB) | CPU only | 19.3 tok/s |
| Qwen3-30B-A3B Q3_K_M (14.7 GB) | 22 of 48 layers in VRAM, rest in RAM (CUDA) | **27.1 tok/s** |

**Speculative decoding**, same laptop. Qwen3.6-27B MTP Q4_K_M (Unsloth, dense, 17 GB), 16K context, 8-bit context
memory, 23 of 64 layers in VRAM and the rest in RAM (CUDA):

| Guesses per check | Generation |
|---|---|
| Off | 3.0 tok/s |
| 2 | 3.9 tok/s |
| **3 (the default)** | **5.3 tok/s** (2.6 tokens per check, 53% of guesses kept) |
| 4 | 4.1 tok/s |

A dense model with most of its weights in RAM is limited by how fast RAM can be read; checking several guessed
tokens per read makes it about 1.75× faster here, with the same answers. More guesses isn't always better: each extra
guess is less likely to be kept but still has to be checked.

**Mixture-of-experts on the same laptop.** Qwen3.6-35B-A3B MTP UD-Q3_K_XL (Unsloth, 17 GB), 16K context (CUDA),
measured by **Find the fastest settings** (short replies at temperature 0, so real chats come out somewhat lower):

| Setting | Without CUDA graphs | 0.5.0 |
|---|---|---|
| Whole layers | 13.2 tok/s | 13.0 tok/s |
| Experts in RAM | 21.7 tok/s | **33.3 tok/s** |
| Experts in RAM, busiest experts in VRAM (hot experts) | 21.0 tok/s | 29.7 tok/s |
| Experts in RAM + speculative decoding, 1 guess per check | 27.5 tok/s | **35.3 tok/s** |
| Experts in RAM + speculative decoding, 2 guesses | 28.3 tok/s | 35.1 tok/s |

Keeping attention on the GPU and only the experts in RAM means the CPU reads just the 8 experts each token picks.
CUDA graphs cut the GPU's share of each token from about 22 ms, mostly spent launching small jobs, to a fraction of
that. Short guesses suit MoE models: every guess the model checks brings in its own experts from RAM. Back-to-back
runs on a laptop vary by 1–2 tok/s with temperature.

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

On Linux? See [Build from source (Linux)](#build-from-source-linux) below: a few commands, and the same web page.

**tierlm's point is running models bigger than your GPU.** It fills GPU memory first, then puts the rest of the
model in system RAM, automatically. On a laptop RTX 3070 with 8 GB, the 14.7 GB Qwen3-30B-A3B runs at about
27 tokens/second.

**What you need**

- **Windows 10 or 11, 64-bit**, on any 64-bit Intel or AMD processor. tierlm picks the fastest code your CPU
  supports when it starts, from plain x86-64 up to AVX2, AVX-512 (Ryzen 7000/9000, Intel workstation chips) and AMX.
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

1. From [Releases](https://github.com/EHTi-Chad/tierlm/releases), download one of the two zips and extract it:
   - **`tierlm-<version>-windows.zip`** (about 550 MB): everything, including NVIDIA's CUDA engine, the fastest
     option on NVIDIA GeForce RTX cards. Works on every PC. Get this one if you have an NVIDIA RTX card.
   - **`tierlm-<version>-windows-lite.zip`** (about 20 MB): the same app without CUDA. Runs on Intel Arc, AMD
     Radeon and NVIDIA cards through Vulkan (slower than CUDA on NVIDIA), or on the CPU.

   tierlm finds your graphics card and picks the fastest engine it has by itself. Nothing else to install: your
   normal graphics driver is enough.
2. Get a model. tierlm uses **GGUF** files, the same ones LM Studio uses; if you already have LM Studio models,
   tierlm can use that folder directly. To download one, pick a size for your computer:
   - Bigger than your GPU, the real test: [`Qwen3-30B-A3B` Q3_K_M](https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF/tree/main)
     (14.7 GB; needs 32 GB RAM with an 8 GB GPU)
   - The newest: [`Qwen3.6-35B-A3B`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main) Q4_K_M (about 21 GB) or
     Q8_0 (for 64 GB RAM). The Q8_0 runs at about 36 tokens/second on an RTX 5090 with 64 GB RAM.
   - Just to check it works: [`Llama-3.2-1B-Instruct` Q4_K_M](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/tree/main) (0.8 GB)

   Put the file in tierlm's `models` folder, or leave it where it is and choose that folder in step 3.
3. Double-click `start-tierlm.bat`. Your browser opens at http://127.0.0.1:8040, where tierlm shows the
   hardware it found and the models in your folder (the **⋯** menu changes the folder). Pick a model and chat.

Step-by-step guide: **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** or the [PDF guide](docs/tierlm-0.5.0-guide.pdf) · Problems: [docs/FAQ.md](docs/FAQ.md)

### Supported models

| Family | Tested on real weights |
|---|---|
| OLMoE | OLMoE-1B-7B-0924-Instruct |
| Qwen3, Qwen3 MoE | Qwen3-0.6B, Qwen3-30B-A3B |
| Llama 3.x | Llama-3.2-1B-Instruct |
| Llama 2 / Mistral 7B / TinyLlama (SentencePiece) | TinyLlama-1.1B-Chat |
| Mixtral, Qwen2/2.5 | model math verified, real weights not yet tested |
| Qwen3.5 / 3.6 / 3.8 dense (`qwen35`, hybrid Gated DeltaNet) | Qwen3.6-27B (Unsloth MTP Q4_K_M) on an RTX 3070 laptop, with speculative decoding |
| Qwen3.5 / 3.6 MoE (`qwen35moe`, e.g. Qwen3.6-35B-A3B) | Qwen3.6-35B-A3B Q8_0 on an RTX 5090 |
| Qwen3-Next / Qwen3-Coder-Next (`qwen3next`, 80B-A3B) | model math verified, real weights not yet tested |

Not yet: Gemma, Phi, DeepSeek, gpt-oss and others. The setup screen says which
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

## Build from source (Linux)

tierlm builds and runs on Linux too (its tests run there). A ready-made Linux download is coming; until then, on
Ubuntu or Debian:

```bash
sudo apt install -y build-essential cmake git ninja-build
sudo apt install -y libvulkan-dev glslc                  # for Vulkan (Intel, AMD, NVIDIA)
git clone --recurse-submodules https://github.com/EHTi-Chad/tierlm.git ~/tierlm
cd ~/tierlm
cmake -S . -B build -G Ninja -DTIERLM_VULKAN=ON         # or -DTIERLM_CUDA=ON with the CUDA toolkit installed
cmake --build build
./build/tierlm-server --models ~/models --port 8040
```

Then open http://127.0.0.1:8040. Details, CUDA versions and WSL notes: [docs/BUILD.md](docs/BUILD.md#linux-and-wsl-cpu-vulkan-or-cuda).

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

- **[docs/tierlm-0.5.0-guide.pdf](docs/tierlm-0.5.0-guide.pdf)**: the full guide with diagrams and test results
- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md): download, run, and use tierlm (no programming needed)
- [docs/FAQ.md](docs/FAQ.md): questions and troubleshooting
- [docs/BUILD.md](docs/BUILD.md): building on Windows and Linux, GPU backends, making a release
- [docs/TOOLS.md](docs/TOOLS.md): every tool and option, environment variables, tests
- [docs/API.md](docs/API.md): the server's HTTP API and event stream
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how tierlm works, and design decisions
- [docs/GGML_CHANGES.md](docs/GGML_CHANGES.md): every change tierlm makes to ggml, why, and what we learned about
  ggml (CUDA graphs, expert ids on CUDA), with a patch file for other projects

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
- [ ] Real-model testing of the new families; more families (Gemma 3/4, Phi, DeepSeek, gpt-oss)
- [x] Qwen3-Next / Qwen3-Coder-Next (both GGUF layouts), verified against a NumPy reference
- [x] Tier loader, first version: models bigger than VRAM split between VRAM and RAM (whole layers or experts), VRAM limit setting
- [ ] SSD tier (stream rarely used weights from disk)
- [ ] Search and download GGUF models from Hugging Face inside tierlm
- [x] OpenAI-compatible API, with an Allow other devices switch and optional API key
- [ ] Tool (function) calling in the API, for coding agents (Cursor agent mode, Cline, Continue agent)
- [x] NVIDIA (CUDA) testing on an RTX 5090
- [ ] AMD GPU testing via Vulkan/ROCm
- [x] Several GPUs as VRAM tiers; auto-tuning of the speed settings; new chat layout with console, chats and benchmark
- [x] Hot experts: per-model expert usage learned from use, the busiest experts kept in VRAM
- [x] CUDA graphs: each token's GPU work replayed with one launch (about 50% faster generation than 0.4.0 on a laptop RTX 3070)
- [ ] Find the fastest settings measured like a real chat (your sampling settings, longer replies)
- [ ] CPU and GPU working on the same layer at once (the VRAM and RAM experts of a layer in parallel; needs changes to ggml's scheduler)
- [x] Speculative decoding with the built-in MTP draft layer; 8-bit and 4-bit KV cache
- [ ] Prompt-lookup drafting (free guesses for code and edits); activation sparsity for dense models

## Credits and licence

- Code: written by Claude (Anthropic), directed and tested by Chad Jameson (EHTi).
- tierlm is released under the [MIT License](LICENSE). Copyright (c) 2026 Chad Jameson (EHTi).
- Third-party components (ggml, cpp-httplib, nlohmann/json) keep their own MIT licences; see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Model weights are not included and have their own licences.
