# Getting started with tierlm

This guide gets you from nothing to chatting with a model in your browser. No programming needed.

tierlm is an early-stage project: it runs the most common model families (Llama, Mistral, Qwen up to Qwen3.8,
OLMoE) but not yet everything on Hugging Face. See [Supported models](#supported-models) before downloading anything.

## What you need

- **Windows 10 or 11, 64-bit**, on any 64-bit Intel or AMD processor. tierlm uses the fastest instructions your
  CPU has (the hardware card says which: AVX-512 on a Ryzen 7000/9000, AVX2 on most PCs from 2015 on).
- **System RAM: at least 32 GB** to run models bigger than your GPU, which is what tierlm is for; **64 GB or
  more is recommended**. The model must fit in GPU memory and RAM together, with about 8 GB to spare for Windows
  and your other apps (8 GB GPU + 32 GB RAM: models up to about 30 GB; 8 GB + 64 GB: about 60 GB; 32 GB + 64 GB:
  about 85 GB). Small models that fit in the GPU run with less.
- **Disk space** for the models you download (1 to 60 GB each) plus about 1 GB for tierlm.
- **Optional: a GPU** for much faster replies: NVIDIA GeForce, Intel Arc (including the built-in Arc graphics in
  Core Ultra laptops) or AMD Radeon. tierlm picks the fastest engine it has by itself: CUDA on NVIDIA RTX cards
  (in the full download), Vulkan on the rest. Without a GPU, tierlm runs on the CPU, which is slower but works.
- Keep your **graphics driver up to date**; Vulkan support comes with it.

On Linux, or to build it yourself, see [BUILD.md](BUILD.md).

## Step 1: Download tierlm

1. Go to the [Releases page](https://github.com/EHTi-Chad/tierlm/releases).
2. Download one of the two zips:
   - **`tierlm-<version>-windows.zip`** (about 550 MB): everything, including NVIDIA's CUDA engine, the fastest
     option on NVIDIA GeForce RTX cards. Get this one if you have an NVIDIA RTX card.
   - **`tierlm-<version>-windows-lite.zip`** (about 55 MB): the same app without CUDA, for Intel Arc, AMD Radeon,
     or no graphics card. It also runs on NVIDIA cards, through Vulkan, just slower.
3. Right-click the zip, choose **Extract All...**, and extract it somewhere simple, such as `C:\tierlm`.

## Step 2: Download the model

1. Open [bartowski/OLMoE-1B-7B-0924-Instruct-GGUF](https://huggingface.co/bartowski/OLMoE-1B-7B-0924-Instruct-GGUF/tree/main)
   on Hugging Face.
2. Download **`OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf`** (about 4.2 GB).
3. Move it into the **`models`** folder inside your tierlm folder, for example
   `C:\tierlm\models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf`. If you already keep models somewhere else
   (LM Studio's folder, say), leave it there: you can point tierlm at that folder in step 4.
4. Big models often come as **several files**, named like `...-00001-of-00003.gguf`, `...-00002-of-00003.gguf`.
   Download **all** of them into the same folder. tierlm shows them as one model ("3 files") and loads them
   together.

## Step 3: Start it

1. Double-click **`start-tierlm.bat`**.
2. The first time, Windows may show **"Windows protected your PC"**. tierlm isn't signed with a paid
   certificate, so Windows doesn't recognise it yet. Click **More info**, then **Run anyway**. (If you want to
   check the download first, compare its SHA-256 with the `.sha256` file on the Releases page.)
3. A black window opens. Leave it open while you use tierlm: it **is** tierlm.
4. Your browser opens **http://127.0.0.1:8040** automatically. If it doesn't, open that address yourself.
5. If Windows Firewall asks about network access, you can click **Cancel**: tierlm only listens on your own
   computer.

## Step 4: Choose your model

The setup screen shows:

1. **Detected hardware**: every GPU tierlm found, plus the CPU, with free memory. You don't choose: tierlm
   uses the GPU if the model fits there (a dedicated GPU first, then integrated graphics), otherwise the CPU,
   and marks it **Will be used**. Below them, a **VRAM** bar shows how the GPU's memory will be used: the
   **model**, its **context** (the KV cache, which grows with the context length you choose), the engine's
   **working memory**, what **other programs** already use, and what's left **free**. If the model doesn't all
   fit, a **System RAM** bar shows the part that goes there. If other programs are using a lot of VRAM (another
   tierlm window, LM Studio, a game), tierlm says so: close them and click **Rescan**. Integrated
   graphics, such as Intel Arc in a laptop, are labelled **Shares system RAM**: they work well, but use the
   computer's normal memory.
2. **GGUF models**: the `.gguf` files in your models folder and its subfolders. Supported ones can be
   selected; others say why they can't be used yet. The **⋯** menu at the top right lets you:
   - **Choose models folder…**: point tierlm at a folder where your models already are. If you use LM Studio,
     tierlm offers its folder directly.
   - **Include subfolders**: on by default.
   - **Find GGUF models on Hugging Face**: opens Hugging Face in your browser.
3. **Settings**: **context length** (how much of the conversation the model remembers; longer uses more
   memory), **context memory precision** (16-bit normally; **8-bit** halves the context's memory with practically
   the same quality, which can let more of the model fit in VRAM; 4-bit quarters it at some cost in quality) and
   **CPU threads**, and **Speculative decoding** for models that include a draft layer (most Qwen3.5 / 3.6 / 3.8
   files, or with a matching `mtp-*.gguf` file in the same folder; the model card says "Speculative decoding"): the
   draft layer guesses a few words ahead and the model checks them all at once, for faster replies with the same
   answers. It's on by default when available; the chat shows how many tokens each check produced. **Advanced** has more: flash attention, prompt batch size,
   active experts per token, a seed for repeatable replies, how to split a model that doesn't fit in VRAM, a
   **VRAM limit**, and **Force CPU**.

Click **Load and start chatting**. Loading takes a few seconds; the first time on a GPU it can take a little
longer while the graphics driver prepares itself. tierlm remembers the folder, and your settings for each
model, in `tierlm-settings.json`. **Change setup** (top right) brings this screen back.

## Step 5: Chat

Type a message and press **Enter**, or click one of the suggestion buttons. The reply streams in word by word.

The right-hand side is the live dashboard:

- **Generation speed**: tokens per second (a token is roughly ¾ of a word), and a chart of speed as the
  conversation gets longer.
- **Context window**: how much of the model's memory of this conversation is used. **New chat** clears it.
- **Where the model lives**: whether the model is on your GPU or CPU, and how much memory it uses.
- **Expert activity map**: the model has 64 "experts" in each of its 16 layers, and picks 8 per layer for
  every token. Each cell is one expert; they flash as they're picked, and warm up the more they're used. Hover
  over a cell for details.
- **Hot-expert cache preview**: an estimate of how much a cache of the most-used experts would help, based on
  your own conversation so far.

**Temperature** controls how adventurous replies are: 0 always picks the most likely word, higher values are
more varied. **Max reply** caps the length of each reply; the default, **No limit**, lets the model finish (thinking models such as Qwen3.x often use thousands of tokens before they answer). **Stop** ends a reply early.

## Step 6: Stop it

Close the black tierlm window, or press **Ctrl+C** in it.

## Connect other apps

tierlm speaks the same API as OpenAI, so apps made for OpenAI (or for LM Studio and Ollama) can use the model
you've loaded. On the setup screen (**Change setup**, top right, from the chat), scroll to **Connect other
apps**: it shows the address to give them, the **Allow other devices** switch, and the **API key**.

- **Apps on this computer** use `http://127.0.0.1:8040/v1` as the base URL, and any text as the API key unless
  you've set one.
- **Other computers and phones** need **Allow other devices** on, then use the second address shown, such as
  `http://192.168.1.20:8040/v1`. If Windows asks whether to let tierlm through the firewall, choose **Private
  networks**. Set an API key, or anyone on your network can use your GPU.

**Open WebUI.** In Open WebUI, open **Admin Panel → Settings → Connections**, add an **OpenAI API** connection with
URL `http://host.docker.internal:8040/v1` if Open WebUI runs in Docker on this computer (turn on **Allow other
devices**: Docker's traffic doesn't count as this computer), or tierlm's network address if it runs elsewhere,
and the API key if you set one. tierlm's models appear in Open WebUI's model list; picking one that isn't loaded
loads it.

**Continue (VS Code / JetBrains).** Add a model to Continue's `config.yaml`:

```yaml
models:
  - name: tierlm
    provider: openai
    model: any            # the model loaded in tierlm
    apiBase: http://127.0.0.1:8040/v1
    apiKey: none          # or your tierlm API key
```

**Cursor.** Cursor sends requests for custom models from its own servers, so it can't reach `127.0.0.1`: tierlm
needs a public HTTPS address from a tunnel, such as Cloudflare's `cloudflared tunnel --url http://localhost:8040`.
Turn on **Allow other devices** and **set an API key** first (anyone who finds the tunnel's address could
otherwise use your GPU). In Cursor's **Settings → Models**, enter the API key, set **Override OpenAI Base URL**
to the tunnel's address plus `/v1`, and add the model name. Chat works; Cursor's agent mode needs tool calling,
which tierlm doesn't have yet.

Details: [API.md](API.md#openai-compatible-api).

## Supported models

tierlm reads the architecture from the file. The setup screen marks each model as supported or says why not.

| Family | GGUF architecture | Examples | Status |
|---|---|---|---|
| OLMoE | `olmoe` | OLMoE-1B-7B-0924-Instruct | Tested on real hardware |
| Qwen3 | `qwen3` | Qwen3-0.6B ... Qwen3-32B | Tested: Qwen3-0.6B |
| Qwen3 MoE | `qwen3moe` | Qwen3-30B-A3B, Qwen3-235B-A22B | Tested: Qwen3-30B-A3B Q3_K_M on an RTX 3070 laptop |
| Qwen2 / 2.5 | `qwen2` | Qwen2.5-7B-Instruct, Qwen2.5-Coder | Model math verified, real-model test pending |
| Llama 3 / 3.1 / 3.2 / 3.3 | `llama` | Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct | Tested: Llama-3.2-1B |
| Llama 2, Mistral 7B, Mixtral, TinyLlama | `llama` | Mistral-7B-Instruct, Mixtral-8x7B | Tested: TinyLlama-1.1B; Mistral and Mixtral math verified |
| Qwen3.5 / 3.6 / 3.8 dense (hybrid) | `qwen35` | Qwen3.8-27B, Qwen3.5-9B | Tested: Qwen3.6-27B on an RTX 3070 laptop, with speculative decoding |
| Qwen3.5 / 3.6 MoE (hybrid) | `qwen35moe` | Qwen3.6-35B-A3B, Qwen3.5-122B-A10B | Tested: Qwen3.6-35B-A3B Q8_0 on an RTX 5090 |
| Qwen3-Next (hybrid MoE) | `qwen3next` | Qwen3-Coder-Next-80B-A3B, Qwen3-Next-80B-A3B | Model math verified, real-model test pending |
| Gemma, Phi, DeepSeek, GLM, gpt-oss, ... | various | | Not yet |

A `mmproj-*.gguf` file next to a model is its **vision add-on** (for reading images). tierlm lists it but
doesn't use it yet.

The OLMoE file used in this guide is the safe choice to start with.

## Options

`start-tierlm.bat` uses port 8040. To use another port, run it from a command prompt as
`start-tierlm.bat 8050`.

You can also run the server directly from a command prompt in the tierlm folder. Give it a model to skip the
setup screen:

```bat
tierlm-server.exe models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf --port 8040 -c 4096 -d Vulkan0
```

| Option | Meaning |
|---|---|
| `--port N` | Web page / API port |
| `-c N` | Context length in tokens (default 4096). Smaller uses less memory |
| `-d auto\|cpu\|<device>` | Which device to use: `auto` picks the best GPU, `cpu` forces the CPU, or a name such as `Vulkan0` (`tierlm-devices.exe` lists them) |
| `--models <folder>` | Where to look for models (default: the folder chosen on the setup screen, else `models`) |
| `-t N` | CPU threads |

Other tools in the folder: `tierlm-devices.exe` (shows which GPU tierlm can use), `tierlm-bench.exe` (speed
test), `tierlm-run.exe` (one-off generation in the command prompt). See [TOOLS.md](TOOLS.md).

## Something not working?

See the [FAQ](FAQ.md).
