#include "model_map.h"
#include "gguf.h"
#include "ggml.h"
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>

using namespace tierlm;
static double gib(uint64_t b) { return b / (1024.0 * 1024.0 * 1024.0); }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s model.gguf [--tensors | --all]\n", argv[0]); return 1; }
    // --all: every setting in the file and every tensor with its shape (to check a new kind of model's layout)
    if (argc > 2 && std::strcmp(argv[2], "--all") == 0) {
        ggml_context* c = nullptr;
        gguf_init_params ip{ true, &c };
        gguf_context* g = gguf_init_from_file(argv[1], ip);
        if (!g) { std::fprintf(stderr, "error: can't read %s\n", argv[1]); return 1; }
        for (int64_t k = 0; k < gguf_get_n_kv(g); ++k) {
            const char* key = gguf_get_key(g, k);
            const gguf_type t = gguf_get_kv_type(g, k);
            std::printf("%-56s ", key);
            switch (t) {
                case GGUF_TYPE_UINT32: std::printf("%u\n", gguf_get_val_u32(g, k)); break;
                case GGUF_TYPE_INT32:  std::printf("%d\n", gguf_get_val_i32(g, k)); break;
                case GGUF_TYPE_FLOAT32: std::printf("%g\n", gguf_get_val_f32(g, k)); break;
                case GGUF_TYPE_BOOL: std::printf("%s\n", gguf_get_val_bool(g, k) ? "true" : "false"); break;
                case GGUF_TYPE_UINT64: std::printf("%llu\n", (unsigned long long)gguf_get_val_u64(g, k)); break;
                case GGUF_TYPE_STRING: { std::string v = gguf_get_val_str(g, k); if (v.size() > 200) v = v.substr(0, 200) + "... (" + std::to_string(v.size()) + " chars)";
                                         for (char& ch : v) if (ch == '\n') ch = ' '; std::printf("%s\n", v.c_str()); break; }
                case GGUF_TYPE_ARRAY: {
                    const gguf_type at = gguf_get_arr_type(g, k); const size_t n = gguf_get_arr_n(g, k);
                    std::printf("[%zu x %s]", n, gguf_type_name(at));
                    if ((at == GGUF_TYPE_INT32 || at == GGUF_TYPE_UINT32) && n <= 64) { for (size_t i = 0; i < n; ++i) std::printf(" %d", ((const int32_t*)gguf_get_arr_data(g, k))[i]); }
                    else if (at == GGUF_TYPE_BOOL && n <= 128) { for (size_t i = 0; i < n; ++i) std::printf("%d", ((const int8_t*)gguf_get_arr_data(g, k))[i] ? 1 : 0); }
                    else if (at == GGUF_TYPE_STRING && n > 0) std::printf(" first: %s", gguf_get_arr_str(g, k, 0));
                    std::printf("\n"); break;
                }
                default: std::printf("(%s)\n", gguf_type_name(t));
            }
        }
        std::printf("\n");
        for (int64_t k = 0; k < gguf_get_n_tensors(g); ++k) {
            const char* name = gguf_get_tensor_name(g, k);
            const ggml_tensor* t = ggml_get_tensor(c, name);
            std::printf("%-48s %-8s [%lld, %lld, %lld, %lld]\n", name, ggml_type_name(t->type), (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
        }
        gguf_free(g); if (c) ggml_free(c);
        return 0;
    }
    ModelMap m;
    try { m = load_model_map(argv[1]); }
    catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }

    if (m.parts > 1) std::printf("split model: %d files\n", m.parts);
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
