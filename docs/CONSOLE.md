# Running tierlm from the console

tierlm's chat page is one way to use it. The other is to start it from a console window with the model and its
settings given as options: it loads the model at once, opens no browser and no page, and serves the API for the
front end you already use (Open WebUI, SillyTavern, a coding agent, your own scripts). A browser tab takes hundreds
of MB of RAM; on a PC where every GB goes to the model, that's worth having back.

- [Quick start: the batch file](#quick-start-the-batch-file)
- [Straight from the command line, Windows or Linux](#straight-from-the-command-line-windows-or-linux)
- [Step by step: Qwen3.8 Flash-Next on one RTX 5090](#step-by-step-qwen38-flash-next-on-one-rtx-5090)
- [Which setup do you want?](#which-setup-do-you-want)
- [Every option](#every-option)
- [Tune once on the page, then start from the console](#tune-once-on-the-page-then-start-from-the-console)
- [Recipes](#recipes)
- [Connecting your front end](#connecting-your-front-end)
- [Problems](#problems)

## Quick start: the batch file

The release zip has `start-tierlm-console.bat` next to `tierlm-server.exe`.

1. Right-click `start-tierlm-console.bat` and choose **Edit** (Notepad opens it).
2. On the `set "MODEL=` line, put your model: the full path of the `.gguf` file (the first file, for a model in
   several files), or its name as tierlm's page lists it in the models folder.
3. Change `CTX` (the context length) and `OPTIONS` if you like (see [Every option](#every-option)); save.
4. Double-click the batch file. When the window says **tierlm is ready**, point your app at
   `http://127.0.0.1:8040/v1`.

Close the window (or press Ctrl+C in it) to stop tierlm. Opening `http://127.0.0.1:8040/v1` in a browser shows a
page saying tierlm is running and what to put in your app; that's also how to check from another device that it
can reach tierlm.

All the settings are together at the top of the file, each explained, with ready-made examples for common setups under them (see also [Which setup do you want?](#which-setup-do-you-want)). The settings:

| Line | What it is |
|---|---|
| `MODEL` | the model: a `.gguf` file, or its name in the models folder |
| `SAVED` | `yes`: start from the settings saved for this model on the page (see below); the other lines still win where filled in |
| `CTX` | context length in tokens, per connection (`32768` = 32K); empty with `SAVED=yes` keeps the saved one |
| `OPTIONS` | any other options from the table below, e.g. `--experts follow --guesses 2 --threads 6` |
| `HOST` | empty: as set on the page (**Allow other devices** in the Connect other apps card); `127.0.0.1`: this computer only; `0.0.0.0`: other devices on your network too |
| `PORT` | the port (8040) |
| `PAGE` | `yes` (default): whoever opens this PC's address in a browser gets tierlm's chat page (it costs this PC nothing until then); `no`: the API only. No browser opens on this PC either way |
| `KEY` | an API key other devices must send; empty: none |

Or skip the batch file and run `tierlm-server.exe` in a console yourself (Command Prompt or PowerShell, in the
tierlm folder):

```
tierlm-server.exe C:\models\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf --ctx 32768 --experts follow --guesses 2 --no-web
```

## Straight from the command line, Windows or Linux

No batch file needed: `tierlm-server` takes the model and every setting as arguments, the same on both systems,
so it fits in your own scripts, services and terminals. `tierlm-server --help` lists every option.

Windows (PowerShell or Command Prompt, in the tierlm folder):

```
.\tierlm-server.exe C:\models\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf --ctx 32768 --experts follow --guesses 2 --threads 6 --no-web
```

Linux (built from source, see the README):

```bash
./build/tierlm-server ~/models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf --ctx 32768 --experts follow --guesses 2 --threads 6 --no-web
```

Things a script can count on:

- **One model, loaded at start.** tierlm loads it before it starts listening; a model that can't be loaded, or
  an option it doesn't know, is said on the error output and tierlm exits with code 1.
- **Ready when `/v1` answers.** `GET /v1` returns `{"object":"status","running":true,...}` once tierlm is up, without
  the API key, so a script can wait for it:
  `until curl -sf http://127.0.0.1:8040/v1 >/dev/null; do sleep 1; done` (Linux), or the same loop with
  `curl.exe` on Windows.
- **The log is the error output**, one timestamped line per event (load, each prompt and reply with its speed):
  `2> tierlm.log` to keep it.
- **Starting from the command line never changes the saved settings**; `--saved` reads them.
- **`--host 0.0.0.0`** serves other devices; give `--api-key` with it. Leave out `--no-web` to serve the chat page
  to them too: no browser opens on the machine running tierlm either way.

As a Linux service, with systemd (`/etc/systemd/system/tierlm.service`, then
`sudo systemctl enable --now tierlm`):

```ini
[Unit]
Description=tierlm
After=network-online.target

[Service]
User=you
WorkingDirectory=/home/you/tierlm
ExecStart=/home/you/tierlm/build/tierlm-server /home/you/models/model.gguf --saved --ctx 32768 --host 0.0.0.0 --port 8040 --api-key choose-a-long-key
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

(Anyone who can read the unit file can read the key in it; keep it to an account that only runs tierlm.)

## Step by step: Qwen3.8 Flash-Next on one RTX 5090

Measured on 12 November 2026: RTX 5090 (32 GB), Ryzen 9 9950X3D, 64 GB DDR5-6000, NVMe SSD, Windows 11, tierlm 0.8.8.
Qwen3.8 Flash-Next UD-IQ3_XXS is 77.9 GiB in 4 files: more than the card's VRAM and the PC's RAM together, once its
n-gram table is counted, so tierlm keeps the busiest experts in VRAM, the rest in RAM and the n-gram table on the SSD.

1. **Get the model.** In tierlm's page, **Get models**, or from unsloth's Qwen3.8 Flash-Next GGUF page on Hugging
   Face: the four `Qwen3.8-Flash-Next-UD-IQ3_XXS-0000N-of-00004.gguf` files and the draft file
   `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, all in one folder. The draft file is what lets tierlm guess ahead
   (`--guesses`): tierlm finds it by itself when it's next to the model.
2. **Choose the SSD tier's drive** once, on the page (the model settings: **SSD tier: where the part that doesn't
   fit in RAM goes**), your fastest NVMe drive. It's saved, and the console start uses it too. (Left as it is, the
   model's own drive is used.)
3. **Edit `start-tierlm-console.bat`** (right-click, **Edit**):

   ```
   set "MODEL=G:\Models\unsloth\Qwen3.8-Flash-Next-UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf"
   set "SAVED=no"
   set "CTX=65536"
   set "OPTIONS=--experts follow --guesses 4 --threads 12"
   ```

   That is: 64K of context, the experts in VRAM following the conversation, four guesses ahead with the draft file,
   12 CPU threads (for the experts in RAM). Save.
4. **Double-click it.** The window shows the draft file and the SSD tier being found, then the model loading:

   ```
   speculative decoding: draft block from G:\Models\unsloth\Qwen3.8-Flash-Next-UD-IQ3_XXS\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
   SSD tier: the n-gram table + 0 layers, 28.8 GB, read as needed from G:\tierlm-ssd\...
   hot experts: 10987 of 24576 experts in VRAM, chosen from earlier use (48 layers split between VRAM and RAM)

     tierlm is ready (no chat page: --no-web)
     OpenAI-compatible API: http://127.0.0.1:8040/v1
   ```

   The first start took 2 minutes on this PC: tierlm measured the GPU's working memory, then loaded once more with
   room for it (a one-time step: it's remembered). Later starts load once.
5. **Connect your app** to `http://127.0.0.1:8040/v1` (see [Connecting your front end](#connecting-your-front-end)).

Asked for a scientific calculator in C# (temperature 0.7), twice, 3,000 tokens each:

```
reply: 3000 tokens at 113.4 tok/s (length); speculative: 4.2 tokens per check, 2280 of 2888 guesses kept
reply: VRAM experts did 92% of this reply's expert work
reply: 3000 tokens at 120.8 tok/s (length); speculative: 4.4 tokens per check, 2321 of 2716 guesses kept
reply: VRAM experts did 93% of this reply's expert work
```

**113 to 121 tokens per second** from a 78 GB model, on one consumer card, with no browser open.

## Which setup do you want?

Four common ways to run tierlm from the console, and the lines to set in `start-tierlm-console.bat` for each. In
all four, no browser opens on the PC running tierlm: what a browser would take in RAM goes to the model.

**1. Just this PC, with your own app on it** (Open WebUI Desktop, opencode, a script, on the same PC)

```
set "HOST=127.0.0.1"
set "PAGE=no"
```

Point the app at `http://127.0.0.1:8040/v1`. Nothing is reachable from other devices.

**2. This PC as the AI server, others chat in their browser** (a desktop or workstation serving laptops)

```
set "HOST=0.0.0.0"
set "PAGE=yes"
set "KEY=choose-a-long-key"
```

1. Start it. Under **tierlm is ready** it prints the address for other devices, e.g.
   `Other devices can connect: http://192.168.1.50:8040`. Windows may ask about the firewall: allow it on
   **Private networks**.
2. On another PC, open that address in a browser: tierlm's chat page, with the model already loaded.
3. The page asks for the key the first time.

The page costs the server nothing until someone opens it; the browser runs on their PC.

**3. This PC as the AI server, others use their own apps** (Open WebUI, opencode, SillyTavern, on their PCs)

```
set "HOST=0.0.0.0"
set "PAGE=no"
set "KEY=choose-a-long-key"
```

In each app: an **OpenAI-compatible** connection with the base URL `http://<the address it prints>/v1` and the key.
To check a PC can reach tierlm, open `http://<the address>/v1` in its browser: it says "tierlm is running".
See [Connecting your front end](#connecting-your-front-end) for Open WebUI's settings.

**4. Both: the chat page for some, their own apps for others**

```
set "HOST=0.0.0.0"
set "PAGE=yes"
set "KEY=choose-a-long-key"
```

The page and the API are on the same address and port. Several people at once: add `--connections 2` (or more)
to `OPTIONS`; each connection has its own context of `CTX` tokens, so the memory for context grows with it.

**Leave `HOST` empty** to use what you set on the page (**Allow other devices** in the **Connect other apps**
card): that's handy when you set up the network once on the page and then only start from the console. A key is
optional on a network that's yours alone, but anyone on the network can use the PC's GPU while it's open.

## Every option

`tierlm-server.exe --help` lists them too.

**Server**

| Option | What it does |
|---|---|
| `--port N` | the port to listen on (8080 when started by hand; the batch files use 8040) |
| `--host ADDRESS` | `127.0.0.1` (default): this computer only; `0.0.0.0`: every device on your network |
| `--api-key KEY` | other devices must send this key (apps on this computer don't) |
| `--no-web` | no chat page: the address answers with one line of text, and the API is all there is |
| `--models DIR` | the models folder |
| `--settings FILE` | the settings file (default `tierlm-settings.json` next to tierlm) |

**The model** (given as the first thing after `tierlm-server.exe`)

| Option | What it does |
|---|---|
| `--saved` | start from the settings saved for this model (see below); options given beside it win |
| `-c N`, `--ctx N` | context length in tokens, per connection (4096) |
| `-t N`, `--threads N` | CPU threads (half the cores) |
| `-d NAME`, `--device NAME` | `auto`, `cpu`, `gpu`, or a device as `tierlm-devices` lists it (`CUDA0`, `Vulkan0`) |
| `--experts MODE` | where a MoE model's experts go when it doesn't fit in VRAM: `auto` (tierlm decides), `layers` (whole layers in VRAM, the rest in RAM), `ram` (experts in RAM, everything else in VRAM), `busiest` (the busiest experts in VRAM too), `follow` (the VRAM experts follow the conversation) |
| `--guesses N` | speculative decoding with the model's draft block (its MTP file): N guesses ahead; 0 off |
| `--lookup N` | guess from the conversation's own text, N tokens; 0 off, -1 automatic (default) |
| `--kv BITS` | context memory precision: `32`, `16` (default), `8`, `5` or `4`; fewer bits fit a longer context |
| `--batch N` | prompt tokens read at once |
| `--connections N` | replies written at once (1); each connection has its own context of `--ctx` tokens |
| `--vram-limit MIB` | use at most this much VRAM |
| `--no-ssd` | keep everything in RAM even when an SSD folder is set |
| `--no-flash-attn` | plain attention |
| `--one-gpu` | don't spread the model over several GPUs |
| `--experts-used N` | experts per token (the model's own number unless given) |
| `--no-thinking` | thinking off, for models that can switch it off |
| `--effort LEVEL` | gpt-oss's reasoning effort: `low`, `medium` or `high` |

A mistake in the options, or a model that can't be loaded, is said in the window and tierlm stops (exit code 1).
Starting from the console never changes the saved settings.

## Tune once on the page, then start from the console

Finding the best settings is easier on the page: **Find the fastest settings** tries the ways a model can be
placed on your hardware and keeps the fastest. tierlm saves what it found (with what the page last loaded the
model with) in `tierlm-settings.json`, and `--saved` starts from that:

1. Start tierlm the usual way (`start-tierlm.bat`), choose the model, click **Find the fastest settings**, then
   **Load**. Close tierlm.
2. From then on: `tierlm-server.exe <the model> --saved --no-web`, or `SAVED=yes` in the batch file.

## Recipes

**A MoE model on an 8 GB card** (32 GB of RAM; Qwen3.6-35B-A3B, 17 GB, about 44 tok/s on an RTX 3070 Laptop):

```
tierlm-server.exe C:\models\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf --ctx 32768 --experts follow --guesses 2 --threads 6 --no-web
```

**A model bigger than RAM on a 32 GB card**, with the SSD tier: choose the SSD tier's drive once on the page (it's
saved; otherwise the model's own drive is used), then give the model as usual; what doesn't fit in VRAM and RAM is
read from the SSD. See the step by step above.

**Sharing one PC with others on your network**, with a key:

```
tierlm-server.exe C:\models\model.gguf --saved --host 0.0.0.0 --api-key choose-a-long-key --no-web
```

Other devices use `http://<this PC's address>:8080/v1` and the key; tierlm prints the address when it starts.
Windows may ask whether to let tierlm through the firewall: allow it on private networks.

**Several people at once:** `--connections 2` writes two replies at once. Each connection has its own context, so
the memory for context doubles; replies are each a little slower, together faster.

## Connecting your front end

Any app that speaks OpenAI's API works: give it the base URL `http://127.0.0.1:8040/v1` (or the PC's network
address from another device), the API key if you set one (anything, or empty, if you didn't), and the model it
lists. tierlm also answers Anthropic's Messages API (`/v1/messages`) and OpenAI's Responses API (`/v1/responses`).

**Open WebUI:**

1. **Settings → Connections** (the admin settings, from your avatar): switch on **OpenAI API** and add
   `http://<address>:<port>/v1`. Not the Ollama section: tierlm doesn't speak Ollama's API, and Open WebUI then
   says "failed to get models".
2. Its **built-in tools** (notes, tasks, memory and more) are sent with every message: about 16,500 tokens before
   you type anything, half of a 32K context. If you don't use them, leave them out: in a chat, the sliders icon
   next to the model name → **Advanced Params** → **Function Calling** → **Legacy**; for every chat, the model's
   settings (avatar → Settings → **Models** → edit) → **Capabilities** → untick **Builtin Tools**. tierlm says it
   in a reply when an app's tools take more than a quarter of the context.

**SillyTavern:** API **Chat Completion**, source **Custom (OpenAI-compatible)**, URL `http://<address>:<port>/v1`.

**Coding agents** (opencode, Continue, Cline, aider and others): an OpenAI-compatible provider with the base URL
above. Agents send long instructions and many tools: give them 64K of context or more where your PC allows.

## Problems

- **The window closes at once, or says the model can't be loaded:** the message above the last line says why
  (a wrong path is the usual one; give the full path in quotes if it has spaces).
- **"the prompt is N tokens and the context length is M"**: the app sends more than the context holds. See
  Open WebUI's built-in tools above, give a longer `--ctx`, or let tierlm make conversations fit (on by default,
  in the page's **Connect other apps** card).
- **Another device can't connect:** `--host 0.0.0.0`, the firewall allowed on private networks, and the right
  address (open `http://<address>:<port>/v1` in that device's browser: it says whether tierlm is running).
- **It's slower than on the page:** the page may have loaded it with settings Find the fastest settings found;
  start with `--saved`.
