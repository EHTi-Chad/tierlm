#pragma once
// The ggml compute engines ("backends": CPU, CUDA, Vulkan, ...), whether they are built into tierlm or, in release
// builds, shipped as separate files that are loaded when tierlm starts. Release builds include one CPU engine per
// x86 generation (plain x86-64, SSE4.2, AVX, AVX2, AVX-VNNI, AVX-512, Zen 4, AMX...) and ggml loads the best one
// this computer supports; a GPU engine whose driver is missing (CUDA on a PC without an NVIDIA card) is skipped.
#include <string>

struct ggml_backend;
struct ggml_backend_buffer_type;

namespace tierlm {

// Loads the engine files next to the program (once; does nothing when the engines are built in).
void load_backends();

ggml_backend* cpu_backend_init();
ggml_backend_buffer_type* cpu_buffer_type();
void set_n_threads(ggml_backend* backend, int n_threads);   // for engines that use CPU threads

// The CPU engine in use and the instruction sets it uses, e.g. "AVX2 FMA F16C" or "AVX512 AVX512_VNNI ...".
std::string cpu_features();

// Testing: adds a tierlm-rpc-server at host:port as a device (a stand-in GPU). False if RPC isn't built.
bool add_rpc_server(const char* endpoint);

}
