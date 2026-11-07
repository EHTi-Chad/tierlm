#pragma once
// The ggml compute engines ("backends": CPU, CUDA, Vulkan, ...), whether they are built into tierlm or, in release
// builds, shipped as separate files that are loaded when tierlm starts. Release builds include one CPU engine per
// x86 generation (plain x86-64, SSE4.2, AVX, AVX2, AVX-VNNI, AVX-512, Zen 4, AMX...) and ggml loads the best one
// this computer supports; a GPU engine whose driver is missing (CUDA on a PC without an NVIDIA card) is skipped.
#include <mutex>
#include <string>
#include <vector>

struct ggml_backend;
struct ggml_backend_buffer_type;

namespace tierlm {

// Loads the engine files next to the program (once; does nothing when the engines are built in).
void load_backends();

// NVIDIA cards that this build's CUDA engine has no code for (a generation it wasn't built for). Before the CUDA
// engine starts, tierlm hides them from it, so they show up (and run) only through Vulkan instead of failing.
struct SkippedGpu { std::string name; int major = 0, minor = 0; };
const std::vector<SkippedGpu>& skipped_cuda_gpus();
std::string cuda_built_for();   // e.g. "7.0, 7.5, 8.6, 8.9, 12.0"; empty when not known

ggml_backend* cpu_backend_init();
ggml_backend_buffer_type* cpu_buffer_type();
void set_n_threads(ggml_backend* backend, int n_threads);   // for engines that use CPU threads
void forget_cpu_backend(ggml_backend* backend);             // before freeing a CPU backend that set_n_threads saw

// The CPU engine in use and the instruction sets it uses, e.g. "AVX2 FMA F16C" or "AVX512 AVX512_VNNI ...".
std::string cpu_features();

// With two models loaded (a chat model and an embedding model), one computes at a time: each step holds this
// lock, so the two take turns between steps instead of competing for the GPU.
std::recursive_mutex& device_mutex();

// Testing: adds a tierlm-rpc-server at host:port (or several, comma-separated) as devices (stand-in GPUs).
// False if RPC isn't built.
bool add_rpc_server(const char* endpoints);

}
