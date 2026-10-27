#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace tierlm {

enum class Role { Embedding, Output, Attention, Norm, DenseFFN, Experts, SharedExpert, Router, Other };
const char* role_name(Role r);

struct TensorInfo {
    std::string name;
    int layer = -1;              // -1 = not inside a transformer block
    Role role = Role::Other;
    std::string type;            // ggml type name, e.g. q4_K
    uint64_t bytes = 0;
    uint64_t offset = 0;         // absolute byte offset in the file
    std::vector<int64_t> shape;
};

struct ModelMap {
    std::string path, arch;
    int n_layers = 0, n_experts = 0, n_experts_used = 0;
    uint64_t total_bytes = 0;
    std::vector<TensorInfo> tensors;
};

// Reads GGUF metadata only (no weights). Throws std::runtime_error on failure.
ModelMap load_model_map(const std::string& path);

}
