# Getting started with tierlm

This guide gets you from nothing to chatting with a model in your browser. No programming needed.

tierlm is an early-stage project: it runs the most common model families (Llama, Mistral, Qwen, OLMoE) but not
yet everything on Hugging Face. See [Supported models](#supported-models) before downloading anything.

## What you need

- **Windows 10 or 11, 64-bit**, with a processor that supports AVX2 (almost every Intel or AMD CPU from 2015
  onwards)
- **About 8 GB of free memory** and **5 GB of disk space**
- **Optional: a GPU with Vulkan support** for faster replies. Intel Arc (including the built-in Arc graphics in
  Core Ultra laptops), AMD Radeon and NVIDIA GeForce cards all work through Vulkan. Without a GPU, tierlm runs
  on the CPU, which is slower but works.
- Keep your **graphics driver up to date**; Vulkan support comes with it.

On Linux, or to build it yourself, see [BUILD.md](BUILD.md).

## Step 1: Download tierlm

1. Go to the [Releases page](https://github.com/EHTi-Chad/tierlm/releases).
2. Download the latest `tierlm-<version>-windows-x64-vulkan.zip`.
3. Right-click the zip, choose **Extract All...**, and extract it somewhere simple, such as `C:\tierlm`.

## Step 2: Download the model

1. Open [bartowski/OLMoE-1B-7B-0924-Instruct-GGUF](https://huggingface.co/bartowski/OLMoE-1B-7B-0924-Instruct-GGUF/tree/main)
   on Hugging Face.
2. Download **`OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf`** (about 4.2 GB).
3. Move it into the **`models`** folder inside your tierlm folder, for example
   `C:\tierlm\models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf`. If you already keep models somewhere else
   (LM Studio's folder, say), leave it there: you can point tierlm at that folder in step 4.

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
   and marks it **Will be used**. A bar shows where the model will go: GPU memory, RAM or SSD. Integrated
   graphics, such as Intel Arc in a laptop, are labelled **Shares system RAM**: they work well, but use the
   computer's normal memory.
2. **GGUF models**: the `.gguf` files in your models folder and its subfolders. Supported ones can be
   selected; others say why they can't be used yet. The **⋯** menu at the top right lets you:
   - **Choose models folder…**: point tierlm at a folder where your models already are. If you use LM Studio,
     tierlm offers its folder directly.
   - **Include subfolders**: on by default.
   - **Find GGUF models on Hugging Face**: opens Hugging Face in your browser.
3. **Settings**: **context length** (how much of the conversation the model remembers; longer uses more
   memory) and **CPU threads**. **Advanced** has more: flash attention, KV cache precision, prompt batch size,
   active experts per token, a seed for repeatable replies, and **Force CPU**.

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
more varied. **Max tokens** caps the length of each reply. **Stop** ends a reply early.

## Step 6: Stop it

Close the black tierlm window, or press **Ctrl+C** in it.

## Supported models

tierlm reads the architecture from the file. The setup screen marks each model as supported or says why not.

| Family | GGUF architecture | Examples | Status |
|---|---|---|---|
| OLMoE | `olmoe` | OLMoE-1B-7B-0924-Instruct | Tested on real hardware |
| Qwen3 | `qwen3` | Qwen3-0.6B ... Qwen3-32B | New: model math verified, real-model test pending |
| Qwen3 MoE | `qwen3moe` | Qwen3-30B-A3B, Qwen3-235B-A22B | New: model math verified, real-model test pending |
| Qwen2 / 2.5 | `qwen2` | Qwen2.5-7B-Instruct, Qwen2.5-Coder | New: model math verified, real-model test pending |
| Llama 3 / 3.1 / 3.2 / 3.3 | `llama` | Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct | New: model math verified, real-model test pending |
| Llama 2, Mistral 7B, Mixtral, TinyLlama | `llama` | Mistral-7B-Instruct, Mixtral-8x7B | New: model math verified, real-model test pending |
| Qwen3.5 / 3.6 hybrid | `qwen35` | Qwen3.8-27B | Not yet (planned) |
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
