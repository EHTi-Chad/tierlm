#include "model_map.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>

using namespace tierlm;
static double gib(uint64_t b) { return b / (1024.0 * 1024.0 * 1024.0); }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s model.gguf [--tensors]\n", argv[0]); return 1; }
    ModelMap m;
    try { m = load_model_map(argv[1]); }
    catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }

    std::printf("arch: %s | layers: %d | experts: %d (active %d) | tensors: %zu | total: %.2f GiB\n\n",
                m.arch.c_str(), m.n_layers, m.n_experts, m.n_experts_used, m.tensors.size(), gib(m.total_bytes));

    std::map<Role, std::pair<uint64_t, int>> by_role;
    std::map<int, uint64_t> by_layer;
    for (const auto& t : m.tensors) {
        auto& r = by_role[t.role]; r.first += t.bytes; r.second++;
        if (t.layer >= 0) by_layer[t.layer] += t.bytes;
    }

    std::printf("%-14s %8s %10s %7s\n", "role", "tensors", "GiB", "share");
    for (const auto& [role, v] : by_role)
        std::printf("%-14s %8d %10.2f %6.1f%%\n", role_name(role), v.second, gib(v.first),
                    100.0 * v.first / m.total_bytes);

    if (!by_layer.empty()) {
        uint64_t lo = UINT64_MAX, hi = 0;
        for (const auto& [l, b] : by_layer) { if (b < lo) lo = b; if (b > hi) hi = b; }
        std::printf("\nper layer: %.3f to %.3f GiB\n", gib(lo), gib(hi));
    }

    if (argc > 2 && std::strcmp(argv[2], "--tensors") == 0) {
        std::printf("\n");
        for (const auto& t : m.tensors)
            std::printf("%-44s %-14s %-8s %10.2f MiB\n", t.name.c_str(), role_name(t.role),
                        t.type.c_str(), t.bytes / (1024.0 * 1024.0));
    }
    return 0;
}
