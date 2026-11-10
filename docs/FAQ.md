# FAQ and troubleshooting

## Using tierlm

**Do I need a GPU?**
No. tierlm runs on the CPU too; it's just slower. With a GPU that supports Vulkan (Intel Arc, AMD Radeon,
NVIDIA GeForce), replies are typically 1.5 to 2 times faster. The pill at the top-left of the page shows which
device is in use.

**Can I run a model bigger than my VRAM and RAM together?**
Yes, with the SSD tier: the parts used least stay on a drive and are read as they're needed (see
[Getting started](GETTING_STARTED.md#models-bigger-than-your-ram-the-ssd-tier)). Mixture-of-experts models suit it
best, since each word reads only the few experts it picks. It's much slower than RAM; pick an NVMe SSD as its drive
in the setup screen. The answers are the same as with everything in RAM.

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
More cards isn't always faster. Keeping the busiest experts in VRAM works with one card for now, and a much slower
second card can cost more than its memory brings: on an RTX 5090 with an RTX 2070 Super added, Qwen3-Coder-Next ran at
46.6 tok/s on both cards and 66.7 on the 5090 alone (with its busiest experts in VRAM). **Find the fastest settings**
measures both ways and keeps the faster; **Use every graphics card** under **Advanced** on the setup screen sets it
yourself.

**What does "Find the fastest settings" do?**
It loads the model and measures, on your PC, the settings that only change speed: how to split a model that doesn't
fit in VRAM, how many CPU threads to use, and speculative decoding with 1 to 4 guesses per check. It keeps the fastest
and saves it for that model. Answers are the same whichever settings win; only the speed differs. On a laptop RTX
3070 it takes Qwen3.6-35B-A3B from 13 tokens/second (whole layers) to about 35. Since 0.5.1 it measures exactly the
way a chat runs, with your sampling settings, so the speed it shows is the speed you'll see in chats (a long
conversation slows down a little as it grows).

**What are "hot experts"?**
In a mixture-of-experts model only a few experts are picked per word, and some far more often than others. tierlm
counts which ones each model uses as you chat (only the counts, kept next to its settings file), and the next time
that model loads with experts in RAM, it puts the busiest experts in VRAM instead of whole expert layers. The expert
map rings them. It's off by default, because whether it's faster depends on the PC: on our RTX 3070 laptop the
experts in VRAM did 71% of the expert work, but the plain split was still a little faster. **Find the fastest
settings** learns the experts right away, measures both, and turns it on only if it wins on your PC. You can also tick
**Keep the busiest experts in VRAM** under **Advanced** on the setup screen yourself, and reset what it learned in
**Model info**. With **Let the VRAM experts follow the conversation** also on, the experts in VRAM keep changing while
you chat: ones the conversation keeps using move in, trading places with ones it stopped using (no extra RAM).
On mixed subjects that was 12% faster on an RTX 5090 with Qwen3.8 Flash-Next and 18-25% on our RTX 3070 laptop with
Qwen3.6-35B-A3B; **Find the fastest settings** measures it too.

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
Yes. In **API** (side menu), tick **Allow other devices** and press **New key** to set an API key. The panel then
shows this PC's network address (like `http://192.168.1.20:8040/v1`); give that and the key to the app on the other
computer, or use **Set up an app** there for ready-made settings. If Windows asks about the firewall, choose
**Private networks**. Keep it to your own network: don't forward the port on your router.

**Can several people or apps use it at the same time?**
Yes, they take turns: one reply is written at a time, and other requests wait in line, first come first served (the
chat shows "Waiting in line: 2 requests ahead of this one…"). Nothing fails with "busy". Switching between
conversations is quick, because tierlm keeps up to 8 recent conversations aside instead of reading them again.
To write several replies at once instead, set **Connections** (Advanced on the setup screen); **Find the fastest
settings** recommends how many your PC can handle. On an RTX 5090, 4 at once with Qwen3.6-35B-A3B Q4_K_M write
364 tok/s in all (about 90 tok/s each), against 177 tok/s for one reply.

**What happens when a conversation fills the context?**
The reply stops and tierlm offers **Keep going** or **New chat** (the context card turns amber a little before).
Keep going archives the oldest messages: they stay in the chat and its file, but the model no longer reads them
every time. For each new message, tierlm looks back at the archive and gives the model the earlier passages that
relate to it (by meaning, with an embedding model loaded; otherwise by keywords), and a line under the reply shows
what it looked back at. **Add a summary for the model** has it write a short summary of the archived part, which it
keeps in mind from then on. Apps using the API get an error by default (coding agents shorten their conversations
themselves); **When an app's conversation is too long, drop its oldest messages** in the API panel changes that.

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
conversation was (only the new message is read); regenerating the last reply goes back to just before it; editing
an earlier message means reading the conversation again from the start.

**Replies repeat themselves or go on forever**
Raise **Repeat penalty** (next to Temperature) a little, for example to 1.15, or **Presence penalty** (just below
it): Qwen suggests 1.5 for its models with thinking off. Very small models loop more easily, especially on vague
prompts.

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

**"This model keeps ... GB in RAM, but only ... GB was free when it loaded"**
Other programs (or a model loaded earlier that Windows hasn't let go of yet) hold RAM that this model counts on. It
still loads, but Windows then moves memory to disk and replies run several times slower: Flash-Next on a 64 GB PC
ran at 18 tok/s that way and 74 after a restart. Close other programs, or restart the PC, and load the model again.
The same note appears under Find the fastest settings' result when it measured that way, since its speeds are then
lower than the PC can do. (Since 0.8.7.)

**What is the `tierlm-cache` folder?**
Since 0.8.7, when a conversation starts with a long system message (1,024 tokens or more: a coding agent's
instructions and tools), tierlm saves what the model holds after reading it, so the next conversation that starts
the same way loads it in a fraction of a second instead of reading it again. The folder is next to
`tierlm-settings.json`, holds at most 8 GB (oldest-used files go first), and can be deleted at any time. To change
the limit or turn it off, set `"prompt_cache_gb"` in `tierlm-settings.json` (0 = off) and restart tierlm.

**The setup screen says "No models found"**
Put a supported `.gguf` file in the `models` folder next to `start-tierlm.bat` (the screen shows the exact
folder), or use the **⋯** menu to choose the folder where your models already are, then click **Rescan**. See [Getting started](GETTING_STARTED.md#step-2-download-the-model).

**My model comes in several files (`-00001-of-00003.gguf` ...)**
That's normal for big models: Hugging Face limits files to 50 GB. Put all the parts in the same folder; tierlm
lists the model once, marked "3 files", and loads every part. If one is missing, the setup screen names it.

**My model shows as "not supported yet"**
The label says which architecture (or tokenizer) the file uses that tierlm doesn't support yet, for example
`qwen35` or `gemma3`. A file labelled **Vision file (mmproj)** isn't a model on its own: it's the image part of
the model next to it, and loads with that model.

**Can it read images?**
Yes, with Qwen3.5 and 3.6 models and their `mmproj-….gguf` vision file in the same folder (LM Studio downloads it
with the model). **Model info** shows **Reads images: yes** when it's loaded. Attach, drag or paste an image
into the chat; apps send them through the API as with OpenAI. Other models read text only, and say so when given a
image. PDFs aren't read yet: turn their pages into images first (your document tools may already do that).

**How does tierlm choose between my GPU and CPU?**
Automatically. It uses a dedicated GPU if the model fits in its memory, then integrated graphics, then the CPU
(the one marked **Will be used**). If a laptop has both Intel and NVIDIA graphics, the NVIDIA one is used. To
run on the CPU anyway, open **Advanced** on the setup screen and tick **Force CPU**. A model too big for the
GPU is split automatically: as much as fits goes in GPU memory, the rest in RAM, and what doesn't fit in RAM either
stays on a drive you pick (the SSD tier), read as each word needs it.

**How much RAM do I need?**
At least 32 GB to run models bigger than your GPU, and 64 GB or more is better. The model has to fit in GPU
memory and RAM together, with about 8 GB left for Windows and other apps: an 8 GB GPU with 32 GB of RAM runs
models up to about 30 GB, with 64 GB about 60 GB. Mixture-of-experts models (names with A3B, A10B...) stay fast
with most of the model in RAM; ordinary dense models slow down a lot.

**Which download do I need?**
Both run on every PC; tierlm picks the fastest engine it has when it starts. `tierlm-<version>-windows.zip`
(about 550 MB) includes NVIDIA's **CUDA** engine, the fastest option on GeForce RTX 20/30/40/50 cards and the
Tesla V100 (and other NVIDIA cards from the V100 on): get it if you have one. `tierlm-<version>-windows-lite.zip` (about 20 MB) has only the **Vulkan** engine, which runs on
Intel Arc, AMD Radeon and NVIDIA cards (slower than CUDA on NVIDIA). Without a GPU, either uses the CPU. With the
full download on a PC without an NVIDIA card, the black window may say CUDA found no device; that's expected.

**Can other apps use tierlm, like they use LM Studio or Ollama?**
Yes: tierlm has an OpenAI-compatible API at `http://127.0.0.1:8040/v1`. Open WebUI, Continue, LibreChat and
scripts using the OpenAI library work with it. See [Connect other apps](GETTING_STARTED.md#connect-other-apps).
Coding agents work too (opencode, Cline, Continue's agent mode, and Cursor through a tunnel): tierlm's API does
tool calling since 0.5.2. See [opencode](GETTING_STARTED.md#connect-other-apps).

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

**"Running on Vulkan instead of CUDA" on an NVIDIA card**
The full download's CUDA engine has code for NVIDIA cards from the Tesla V100 (Volta) on. An older card, such as a
GTX 10-series, Tesla P40 or P100 (Pascal), can't run it, so tierlm uses that card through Vulkan instead, which works
but is slower. The setup screen and **tierlm-devices.exe** say when this happens. In a PC with both kinds of card,
the newer ones still use CUDA. (To choose the CUDA cards yourself, set `CUDA_VISIBLE_DEVICES`; tierlm then leaves
the choice to you.)

## Still stuck?

Open an issue on the [GitHub repository](https://github.com/EHTi-Chad/tierlm/issues) with the output of
`tierlm-devices.exe` and the error message.
