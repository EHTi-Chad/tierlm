# Building tierlm

tierlm is a CMake project in C++17. ggml is included as a git submodule; cpp-httplib and nlohmann/json are
single headers in `third_party/`. Every build includes the CPU backend; GPU backends are switched on with
CMake options.

| Option | Default | What it adds |
|---|---|---|
| `-DTIERLM_VULKAN=ON` | off | Vulkan backend: Intel Arc, AMD and NVIDIA GPUs (needs the Vulkan SDK) |
| `-DTIERLM_CUDA=ON` | off | CUDA backend: NVIDIA GPUs (needs the CUDA toolkit) |
| `-DCMAKE_BUILD_TYPE=Release` | Release | Optimised build. Debug builds are roughly 10× slower |
| `-DTIERLM_PORTABLE=ON` | off | Binaries for distribution: ggml's engines as separate files, with one CPU engine per x86 generation (x86-64, SSE4.2, AVX, AVX2, AVX-VNNI, AVX-512, Zen 4, AMX...) of which the best is loaded at start-up; static MSVC runtime |

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

`scripts\package-windows.bat` builds a portable release and zips it. Portable builds (`-DTIERLM_PORTABLE=ON`) run on
every 64-bit Intel or AMD CPU: ggml's engines are separate `ggml*.dll` files next to the programs, with a CPU engine
for each x86 generation; when tierlm starts, ggml loads the best one the processor supports (AVX-512 on a Ryzen 9000,
AVX2 on most PCs from 2015 on, plain x86-64 on anything older) and any GPU engine whose driver is present. The C
runtime is linked in, so users don't need the Visual C++ redistributable. From a Developer Command Prompt in the
project folder:

```bat
scripts\package-windows.bat 0.5.2 both
```

It makes two downloads. **`tierlm-0.5.2-windows.zip`** has both GPU engines, CUDA (for NVIDIA RTX 20/30/40/50)
and Vulkan (Intel, AMD, older NVIDIA), and tierlm picks one at start-up; it runs on every PC.
**`tierlm-0.5.2-windows-lite.zip`** (`lite` on its own) is Vulkan only, about 20 MB. Leave out the second word
to make just the full one. The build machine needs the CUDA toolkit 12.8 or newer
(for RTX 50-series) and the Vulkan SDK. NVIDIA's redistributable runtime DLLs (`cudart`, `cuBLAS`, `cuBLASLt`,
about 700 MB unzipped) are bundled, so users only need an up-to-date graphics driver. NVIDIA's driver DLL
(`nvcuda.dll`) is delay-loaded, so the same program starts on PCs without an NVIDIA card. Without the CUDA toolkit, make
only the `lite` one. They build into `build-release` and `build-release-vulkan`, then write
`dist\tierlm-0.5.2-windows.zip` and `dist\tierlm-0.5.2-windows-lite.zip` (the tools, `web\`, `start-tierlm.bat`, an empty `models\` folder,
the docs and all licences) plus a `.sha256` checksum. Publish
both on a GitHub release, for example from WSL:

```bash
gh release create v0.5.2 /mnt/c/dev/tierlm/dist/tierlm-0.5.2-windows-*.zip* --title "tierlm 0.5.2" --notes "..."
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

# The hybrid Qwen3.5 / 3.8 architecture (Gated DeltaNet), against a NumPy version of Hugging Face's model
~/tierlm-oracle/bin/python tests/tiny_qwen35.py build/tierlm-logits

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

`tiny_models.py` and `tiny_qwen35.py` run two passes per architecture: **exact**, on the CPU with an f32 KV cache,
must match the NumPy reference to about 1e-5; **normal**, with the usual settings on the default device (your GPU
if you have one), must agree within GPU rounding (up to a few %, from the f16 KV cache, f16 flash attention, and the
TF32 tensor-core matrix multiplies CUDA uses for small batches on RTX 30-series and newer) and pick the same top
token. Real bugs show up as errors of 100% or more. Each test prints `PASS` when everything is right.

`scripts\diag-qwen35.bat` and `scripts\diag-qwen35b.bat` show how a GPU difference was tracked down: they move one
operation type at a time to the CPU (`TIERLM_CPU_OPS`) and show which one changes the result.

To regenerate the tokenizer cases or the Unicode tables (only needed when changing them):

```bash
~/tierlm-oracle/bin/python tests/ref_tokenize.py      # writes tests/tok_cases.jsonl
~/tierlm-oracle/bin/python -c "import json; o=open('tests/tok_cases.txt','w'); [o.write(json.loads(l)['text'].encode().hex()+'\t'+' '.join(map(str,json.loads(l)['ids']))+'\n') for l in open('tests/tok_cases.jsonl',encoding='utf-8')]"
~/tierlm-oracle/bin/python tests/gen_unicode.py       # writes src/unicode_tables.h
```
