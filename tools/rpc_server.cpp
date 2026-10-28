// tierlm-rpc-server (testing only): serves this machine's CPU as a ggml RPC device. tierlm sees an RPC
// device as a GPU, so the tests can check that splitting a model between "VRAM" and RAM gives the same
// results, on machines without a GPU.
//   tierlm-rpc-server 127.0.0.1:50052
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include <algorithm>
#include <cstdio>
#include <thread>

int main(int argc, char** argv) {
    const char* endpoint = argc > 1 ? argv[1] : "127.0.0.1:50052";
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    std::fprintf(stderr, "serving the CPU as an RPC device on %s\n", endpoint);
    ggml_backend_rpc_start_server(endpoint, nullptr, std::max(1u, std::thread::hardware_concurrency() / 2), 1, &cpu);
    return 0;
}
