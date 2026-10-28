# Building tierlm

tierlm is a CMake project in C++17. ggml is included as a git submodule; cpp-httplib and nlohmann/json are
single headers in `third_party/`. Every build includes the CPU backend; GPU backends are switched on with
CMake options.

| Option | Default | What it adds |
|---|---|---|
| `-DTIERLM_VULKAN=ON` | off | Vulkan backend: Intel Arc, AMD and NVIDIA GPUs (needs the Vulkan SDK) |
| `-DTIERLM_CUDA=ON` | off | CUDA backend: NVIDIA GPUs (needs the CUDA toolkit) |
| `-DCMAKE_BUILD_TYPE=Release` | Release | Optimised build. Debug builds are roughly 10× slower |
| `-DTIERLM_PORTABLE=ON` | off | Binaries for distribution: AVX2 baseline instead of the build machine's CPU, static MSVC runtime |

Get the source with the submodule:

```bash
git clone --recurse-submodules https://github.com/EHTi-Chad/tierlm.git
# or, in an existing clone:
git submodule update --init
```

The first build compiles ggml, and with Vulkan, several hundred GPU shaders. Expect a few minutes; later
builds only recompile what changed.

---

## Windows (Vulkan, recommended for Intel Arc)

### Requirements

- **Visual Studio 2022 or 2026** (Community is fine) with the **Desktop development with C++** workload. This
  provides the MSVC compiler, CMake and Ninja.
- **Vulkan SDK** from https://vulkan.lunarg.com (installs to `C:\VulkanSDK\<version>` and sets `VULKAN_SDK`).
- An up-to-date GPU driver.

### Build

Open a **Developer Command Prompt** for Visual Studio. If PowerShell blocks scripts on your machine, start it
from PowerShell like this instead:

```powershell
cmd /k "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
```

(Use `...\2022\...` for Visual Studio 2022.) Then:

```bat
cd /d C:\dev\tierlm
cmake -S . -B build -G Ninja -DTIERLM_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
build\tierlm-devices.exe
```

`tierlm-devices` should list your GPU as `Vulkan0` and end with `build type: Release`.

### Windows pitfalls

- **Debug by accident.** With MSVC, CMake may quietly default to a Debug build. tierlm's CMakeLists sets
  Release when nothing is specified, but if an old `build` folder was configured as Debug, delete it
  (`rmdir /s /q build`) and configure again. `tierlm-devices` prints the build type.
- **Vulkan finds no driver (`ERROR_INCOMPATIBLE_DRIVER`, "Found no drivers").** Check for leftover
  environment variables that force a specific driver file:

  ```bat
  set | findstr /i "VK_"
  ```

  If `VK_ICD_FILENAMES` or `VK_DRIVER_FILES` point at a driver folder, they may be stale after a driver
  update. Clearing them fixed this on a Lunar Lake laptop:

  ```bat
  reg delete HKCU\Environment /v VK_ICD_FILENAMES /f
  reg delete HKCU\Environment /v VK_DRIVER_FILES /f
  ```

  Then sign out and back in, or clear them in the current window with `set VK_ICD_FILENAMES=`. The Vulkan
  loader ignores these variables in elevated (admin) processes, so if Vulkan works as administrator but not as
  a normal user, this is the likely cause. `set VK_LOADER_DEBUG=error,warn` makes the loader explain what it
  is doing.
- **First GPU run is slow.** The driver compiles GPU pipelines on first use and caches them; the second run
  starts immediately.
- **Port already in use.** If `http://localhost:8080` shows a different app, another program owns the port.
  Use `--port 8040` (or any free port) and browse to `http://127.0.0.1:<port>`.

---

## Linux and WSL (CPU, Vulkan or CUDA)

### Requirements

```bash
sudo apt install -y build-essential cmake git ninja-build
# Vulkan backend:
sudo apt install -y libvulkan-dev glslc
```

For CUDA, install the CUDA toolkit for your GPU. RTX 50-series (Blackwell, SM120) needs **CUDA 12.8 or newer**.

### Build

```bash
cd ~/tierlm
cmake -S . -B build -G Ninja                       # CPU only
cmake -S . -B build -G Ninja -DTIERLM_VULKAN=ON    # or Vulkan
cmake -S . -B build -G Ninja -DTIERLM_CUDA=ON      # or CUDA
cmake --build build
```

### WSL notes

- **Work in the Linux file system** (`~/...`), not `/mnt/c/...`. Builds on `/mnt/c` are many times slower.
- **WSL cannot see an Intel NPU**, and Vulkan inside WSL goes through a DirectX translation layer. For Intel
  Arc, build natively on Windows instead.
- **Thread count matters.** On laptops with performance and efficiency cores (and inside a VM like WSL),
  using every core can be dramatically slower, because ggml's threads wait for the slowest one hundreds of
  times per token. tierlm defaults to half the hardware threads; try `-t 4` or `-t 2` if speed is poor.
- **Git can't write to `C:` from WSL** ("chmod ... Operation not permitted"): add this to `/etc/wsl.conf`,
  then run `wsl --shutdown` from PowerShell:

  ```ini
  [automount]
  options = "metadata,uid=1000,gid=1000,umask=022"
  ```

---

## Making a Windows release

`scripts\package-windows.bat` builds a portable release and zips it: binaries tuned for any AVX2 CPU instead
of the build machine (`-DTIERLM_PORTABLE=ON`), with the C runtime linked in so users don't need the Visual C++
redistributable. From a Developer Command Prompt in the project folder:

```bat
scripts\package-windows.bat 0.1.0
```

It builds into `build-release`, then writes `dist\tierlm-0.1.0-windows-x64-vulkan.zip` (the tools, `web\`,
`start-tierlm.bat`, an empty `models\` folder, the docs and all licences) plus a `.sha256` checksum. Publish
both on a GitHub release, for example from WSL:

```bash
gh release create v0.1.0 /mnt/c/dev/tierlm/dist/tierlm-0.1.0-windows-x64-vulkan.zip* --title "tierlm 0.1.0" --notes "..."
```

---

## Running the tests

The tests compare tierlm against independent references. They need a small Python environment; it is only
for testing and is not part of the engine.

```bash
python3 -m venv ~/tierlm-oracle
~/tierlm-oracle/bin/pip install tokenizers huggingface_hub gguf numpy

# Model math for every architecture (no real model needed: builds tiny random models)
~/tierlm-oracle/bin/python tests/tiny_models.py build/tierlm-logits

# Tokenizers, without any model: trains small BPE and SentencePiece tokenizers and compares
~/tierlm-oracle/bin/pip install sentencepiece
~/tierlm-oracle/bin/python tests/trained_tokenizers.py build/tierlm-tok-test

# Tokenizers of real models against their official Hugging Face tokenizers
~/tierlm-oracle/bin/python tests/check_real_models.py build/tierlm-tok-test ~/models/Qwen3-0.6B-Q8_0.gguf=Qwen/Qwen3-0.6B

# OLMoE tokenizer (reference IDs are already in tests/tok_cases.txt)
./build/tierlm-tok-test ~/models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf tests/tok_cases.txt
```

To test splitting a model between a GPU and RAM on a machine without a GPU, build with ggml's RPC backend; a
local `tierlm-rpc-server` then stands in for a GPU, and every architecture is checked at 10 different splits:

```bash
cmake -S . -B build-rpc -G Ninja -DGGML_RPC=ON && cmake --build build-rpc
./build-rpc/tierlm-rpc-server 127.0.0.1:50052 &
TIERLM_RPC=127.0.0.1:50052 ~/tierlm-oracle/bin/python tests/tiny_models.py build-rpc/tierlm-logits
```

`tiny_models.py` runs two passes per architecture: with an exact f32 KV cache, results must match the NumPy
reference to about 1e-6; with the normal f16 cache, they must agree within rounding and pick the same top
token. Each test prints `PASS` when everything is right.

To regenerate the tokenizer cases or the Unicode tables (only needed when changing them):

```bash
~/tierlm-oracle/bin/python tests/ref_tokenize.py      # writes tests/tok_cases.jsonl
~/tierlm-oracle/bin/python -c "import json; o=open('tests/tok_cases.txt','w'); [o.write(json.loads(l)['text'].encode().hex()+'\t'+' '.join(map(str,json.loads(l)['ids']))+'\n') for l in open('tests/tok_cases.jsonl',encoding='utf-8')]"
~/tierlm-oracle/bin/python tests/gen_unicode.py       # writes src/unicode_tables.h
```
