# FAQ and troubleshooting

## Using tierlm

**Do I need a GPU?**
No. tierlm runs on the CPU too; it's just slower. With a GPU that supports Vulkan (Intel Arc, AMD Radeon,
NVIDIA GeForce), replies are typically 1.5 to 2 times faster. The pill at the top-left of the page shows which
device is in use.

**Why is a dense model (like Qwen3.6-27B) so much slower than a bigger MoE model when it doesn't fit in VRAM?**
A dense model reads all of its weights for every word. When most of them are in RAM, speed is limited by how fast
RAM delivers them: a laptop's RAM gives roughly 40 GB/s, so 14 GB of weights in RAM means about 3 words per second.
A mixture-of-experts model (names with A3B, A10B...) only reads the few experts each word uses, about 3 billion
parameters for an A3B model, so it stays fast with most of the model in RAM. To speed up a dense model, shorten the
context or set **Context memory precision** to **8-bit** (the context takes VRAM that could hold more layers), or
pick a smaller quantization; better, use an MoE model.

**What is speculative decoding, and why is it greyed out?**
Many Qwen3.5 / 3.6 / 3.8 files include a small "draft" layer trained to guess the next word. With **Speculative
decoding** on, it guesses a few words ahead and the model checks all of them in one pass, so each read of the
model's weights produces more than one word: faster replies, with exactly the answers the model would give anyway.
It's greyed out when the model file has no draft layer. Some publishers (Unsloth) put it in a separate file named
`mtp-<model>.gguf`: download that too and keep it in the same folder, and tierlm uses it automatically (the setup
screen lists it as a **Draft add-on**). The chat shows "tokens per check" after each reply: above about 1.5 it's
helping. **Guesses per check** defaults to 3 for dense models, the fastest on our tests (on a laptop RTX 3070 with Qwen3.6-27B: 3.0 tok/s
off, 3.9 with 2, 5.3 with 3, 4.1 with 4); try 2 or 4 for your model and PC. For mixture-of-experts models the default is 2 (Qwen3.6-35B-A3B on the same laptop:
35 tok/s with 1 or 2, 28 with 3), because each guess the model checks pulls in its own experts.

**Does tierlm use two graphics cards?**
Yes, when they're the same kind (two NVIDIA cards with CUDA, say): the model fills the first card, then the second,
then system RAM. The setup screen shows a VRAM bar for each card. Different kinds (an NVIDIA and an AMD card) aren't
combined yet: tierlm uses the bigger one.

**What does "Find the fastest settings" do?**
It loads the model and measures, on your PC, the settings that only change speed: how to split a model that doesn't
fit in VRAM, how many CPU threads to use, and speculative decoding with 1 to 4 guesses per check. It keeps the fastest
and saves it for that model. Answers are the same whichever settings win; only the speed differs. On a laptop RTX
3070 it takes Qwen3.6-35B-A3B from 13 tokens/second (whole layers) to about 35. It measures with short replies
at temperature 0, so the speeds it shows are a little higher than you'll see in a normal chat.

**What are "hot experts"?**
In a mixture-of-experts model only a few experts are picked per word, and some far more often than others. tierlm
counts which ones each model uses as you chat (only the counts, kept next to its settings file), and the next time
that model loads with experts in RAM, it puts the busiest experts in VRAM instead of whole expert layers. The expert
map rings them. It's off by default, because whether it's faster depends on the PC: on our RTX 3070 laptop the
experts in VRAM did 71% of the expert work, but the plain split was still a little faster. **Find the fastest
settings** learns the experts right away, measures both, and turns it on only if it wins on your PC. You can also tick
**Keep the busiest experts in VRAM** under **Advanced** on the setup screen yourself, and reset what it learned in
**Model info**.

**Why is 0.5.0 so much faster on NVIDIA cards?**
It uses **CUDA graphs**: each token's GPU work is recorded once and then replayed with a single launch, instead of
thousands of small launches. On a laptop RTX 3070, Qwen3.6-35B-A3B with experts in RAM went from 21.7 to 33.3
tokens/second, with exactly the same answers. Intel and AMD cards (Vulkan) and the CPU aren't affected.
[GGML_CHANGES.md](GGML_CHANGES.md) explains what it took.

**Can I use a different model?**
Yes, for the common families: Llama (2, 3.x), Mistral and Mixtral, Qwen2/2.5, Qwen3 and Qwen3 MoE, the Qwen3.5 /
3.6 / 3.8 dense models (such as Qwen3.8-27B) and MoE models (such as Qwen3.6-35B-A3B), Qwen3-Next and
Qwen3-Coder-Next, and OLMoE.
More families are being added. The setup screen says which of your files are supported, and why the others
aren't. See [Supported models](GETTING_STARTED.md#supported-models).

**Does anything leave my computer?**
No. The model runs entirely on your machine, and the web page talks only to `127.0.0.1` (your own computer).
tierlm doesn't collect or send anything.

**Can other computers on my network use it?**
Only if you start the server with `--host 0.0.0.0`. There's no password or login, so only do that on a
network you trust.

**Why do replies get slower in long conversations?**
Every new word has to look back over the whole conversation so far. tierlm uses flash attention to keep this
efficient, but very long chats still slow down. Click **New chat** to start fresh. The speed-vs-context chart
on the dashboard shows the effect.

**Is the model always right?**
No. Small models such as OLMoE-1B-7B (about 1 billion active parameters) are good for experiments and simple
questions, but they make mistakes and can invent things confidently, such as code libraries that don't exist.
Bigger models are much more reliable.

**Qwen3.5 / 3.8: why does editing an earlier message take longer?**
These are hybrid models: most layers keep a fixed-size running memory instead of a KV cache. That's why long
conversations are cheap for them, but the memory can't be rewound. Adding a message continues where the
conversation was; editing or regenerating an earlier message means reading the conversation again from the start.

**Replies repeat themselves or go on forever**
Raise **Repeat penalty** (next to Temperature) a little, for example to 1.15. Very small models loop more
easily, especially on vague prompts.

## Problems

**"Windows protected your PC" when starting it**
tierlm isn't code-signed, so SmartScreen doesn't recognise it. Click **More info**, then **Run anyway**. You
can verify the zip against the `.sha256` file on the Releases page first. Code signing is on the way.

**"Part of this app has been blocked"**
This comes from Windows' Smart App Control, which is on for some Windows 11 PCs. It checks every file a program
loads, and because tierlm isn't code-signed yet, it can refuse one. Since 0.5.0 tierlm only loads the files it
actually uses (before, it also tried every spare CPU engine, and those were what got blocked). If you still see it,
the black window or **tierlm-devices.exe** shows what didn't load; please
[open an issue](https://github.com/EHTi-Chad/tierlm/issues) with the file name. To find it, run this in PowerShell
as administrator:
`Get-WinEvent -LogName "Microsoft-Windows-CodeIntegrity/Operational" -MaxEvents 20 | Format-List TimeCreated, Message`.
Smart App Control can't make an exception for one program; signed releases are the real fix and are on the way.

**The setup screen says "No models found"**
Put a supported `.gguf` file in the `models` folder next to `start-tierlm.bat` (the screen shows the exact
folder), or use the **⋯** menu to choose the folder where your models already are, then click **Rescan**. See [Getting started](GETTING_STARTED.md#step-2-download-the-model).

**My model comes in several files (`-00001-of-00003.gguf` ...)**
That's normal for big models: Hugging Face limits files to 50 GB. Put all the parts in the same folder; tierlm
lists the model once, marked "3 files", and loads every part. If one is missing, the setup screen names it.

**My model shows as "not supported yet"**
The label says which architecture (or tokenizer) the file uses that tierlm doesn't support yet, for example
`qwen35` or `gemma3`. A file labelled **Vision add-on (mmproj)** isn't a model on its own: it's the image part of
the model next to it.

**How does tierlm choose between my GPU and CPU?**
Automatically. It uses a dedicated GPU if the model fits in its memory, then integrated graphics, then the CPU
(the one marked **Will be used**). If a laptop has both Intel and NVIDIA graphics, the NVIDIA one is used. To
run on the CPU anyway, open **Advanced** on the setup screen and tick **Force CPU**. A model too big for the
GPU is split automatically: as much as fits goes in GPU memory, the rest in RAM. (An SSD tier is next.)

**How much RAM do I need?**
At least 32 GB to run models bigger than your GPU, and 64 GB or more is better. The model has to fit in GPU
memory and RAM together, with about 8 GB left for Windows and other apps: an 8 GB GPU with 32 GB of RAM runs
models up to about 30 GB, with 64 GB about 60 GB. Mixture-of-experts models (names with A3B, A10B...) stay fast
with most of the model in RAM; ordinary dense models slow down a lot.

**Which download do I need?**
Both run on every PC; tierlm picks the fastest engine it has when it starts. `tierlm-<version>-windows.zip`
(about 550 MB) includes NVIDIA's **CUDA** engine, the fastest option on GeForce RTX 20/30/40/50 cards: get it if
you have one. `tierlm-<version>-windows-lite.zip` (about 20 MB) has only the **Vulkan** engine, which runs on
Intel Arc, AMD Radeon and NVIDIA cards (slower than CUDA on NVIDIA). Without a GPU, either uses the CPU. With the
full download on a PC without an NVIDIA card, the black window may say CUDA found no device; that's expected.

**Can other apps use tierlm, like they use LM Studio or Ollama?**
Yes: tierlm has an OpenAI-compatible API at `http://127.0.0.1:8040/v1`. Open WebUI, Continue, LibreChat and
scripts using the OpenAI library work with it. See [Connect other apps](GETTING_STARTED.md#connect-other-apps).
Coding agents that need tool calling (Cursor's agent mode, Cline) don't work yet.

**Another computer, my phone or Open WebUI in Docker can't connect**
Turn on **Allow other devices** in **Connect other apps** on the setup screen, and use the network address it
shows (not 127.0.0.1). If Windows asked about the firewall and you clicked Cancel, allow `tierlm-server` in
**Windows Security → Firewall & network protection → Allow an app through firewall**, for Private networks.

**My models are already in another folder (LM Studio, say)**
Click the **⋯** menu next to **GGUF models** and choose **Choose models folder…**, or the LM Studio entry if
tierlm found one. Subfolders are searched too. tierlm remembers the folder.

**The page shows a different app, or "Server unavailable / 404"**
Another program is already using the port. Start tierlm on a different one, for example
`start-tierlm.bat 8050`, and open `http://127.0.0.1:8050`. Use `127.0.0.1` rather than `localhost`, so you
reach tierlm and not something else listening on the same port.

**My GPU isn't listed on the setup screen (or it says `device: CPU`)**
If the setup screen shows a yellow box about `VK_ICD_FILENAMES` or `VK_DRIVER_FILES`, follow it (details
below). Otherwise, run `tierlm-devices.exe` from a command prompt in the tierlm folder. It lists the GPUs tierlm can see and,
if Vulkan fails to start, the reason.

- Update your graphics driver from Intel, AMD or NVIDIA.
- If it reports **"Found no drivers"** or **ERROR_INCOMPATIBLE_DRIVER** even though your driver is current,
  check for leftover settings that point Vulkan at an old driver: open a command prompt and run
  `set | findstr /i VK_`. If `VK_ICD_FILENAMES` or `VK_DRIVER_FILES` appear, remove them:

  ```bat
  reg delete HKCU\Environment /v VK_ICD_FILENAMES /f
  reg delete HKCU\Environment /v VK_DRIVER_FILES /f
  ```

  Then sign out of Windows and back in. (A tell-tale sign: Vulkan works when run as administrator but not as
  your normal account.)

**The first reply on a GPU takes several seconds**
The graphics driver compiles its GPU programs the first time and caches them. Later replies, and later runs,
start immediately.

**It's slow on the CPU**
Laptops with "performance" and "efficiency" cores can be much slower if tierlm uses every core, because all
threads wait for the slowest one. tierlm uses half your threads by default; try `-t 4` or `-t 2` when running
`tierlm-server.exe` directly.

**"not enough device memory"**
Close other programs, or pick a shorter context length on the setup screen. On laptops with integrated
graphics, the GPU shares system memory, so other apps compete for it.

**The black window closes immediately**
Open a command prompt in the tierlm folder and run `start-tierlm.bat` from there, so you can read the error
message.

## Still stuck?

Open an issue on the [GitHub repository](https://github.com/EHTi-Chad/tierlm/issues) with the output of
`tierlm-devices.exe` and the error message.
