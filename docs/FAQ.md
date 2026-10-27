# FAQ and troubleshooting

## Using tierlm

**Do I need a GPU?**
No. tierlm runs on the CPU too; it's just slower. With a GPU that supports Vulkan (Intel Arc, AMD Radeon,
NVIDIA GeForce), replies are typically 1.5 to 2 times faster. The pill at the top-left of the page shows which
device is in use.

**Can I use a different model?**
Not yet. tierlm currently understands one model family, OLMoE. Other GGUF files fail with "expected olmoe".
Qwen3 MoE support is next. See [Supported models](GETTING_STARTED.md#supported-models).

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
No. OLMoE-1B-7B is a small model (about 1 billion active parameters). It's good for experiments and simple
questions, but it makes mistakes and can state wrong things confidently.

## Problems

**"Windows protected your PC" when starting it**
tierlm isn't code-signed, so SmartScreen doesn't recognise it. Click **More info**, then **Run anyway**. You
can verify the zip against the `.sha256` file on the Releases page first.

**"No model found"**
Put a supported `.gguf` file in the `models` folder next to `start-tierlm.bat`. See
[Getting started](GETTING_STARTED.md#step-2-download-the-model).

**The page shows a different app, or "Server unavailable / 404"**
Another program is already using the port. Start tierlm on a different one, for example
`start-tierlm.bat 8050`, and open `http://127.0.0.1:8050`. Use `127.0.0.1` rather than `localhost`, so you
reach tierlm and not something else listening on the same port.

**It says `device: CPU` but I have a GPU**
Run `tierlm-devices.exe` from a command prompt in the tierlm folder. It lists the GPUs tierlm can see and,
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
Close other programs, or start with a smaller context, for example `-c 2048`. On laptops with integrated
graphics, the GPU shares system memory, so other apps compete for it.

**The black window closes immediately**
Open a command prompt in the tierlm folder and run `start-tierlm.bat` from there, so you can read the error
message.

## Still stuck?

Open an issue on the [GitHub repository](https://github.com/EHTi-Chad/tierlm/issues) with the output of
`tierlm-devices.exe` and the error message.
