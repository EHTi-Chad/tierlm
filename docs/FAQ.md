# FAQ and troubleshooting

## Using tierlm

**Do I need a GPU?**
No. tierlm runs on the CPU too; it's just slower. With a GPU that supports Vulkan (Intel Arc, AMD Radeon,
NVIDIA GeForce), replies are typically 1.5 to 2 times faster. The pill at the top-left of the page shows which
device is in use.

**Can I use a different model?**
Yes, for the common families: Llama (2, 3.x), Mistral and Mixtral, Qwen2/2.5, Qwen3 and Qwen3 MoE, the Qwen3.5 /
3.6 / 3.8 dense models (such as Qwen3.8-27B) and MoE models (such as Qwen3.6-35B-A3B), and OLMoE.
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
can verify the zip against the `.sha256` file on the Releases page first.

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
run on the CPU anyway, open **Advanced** on the setup screen and tick **Force CPU**. Splitting a model that is
too big for the GPU between GPU memory, RAM and SSD is the next big feature.

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
