#include "gguf.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 1; }
    gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context* g = gguf_init_from_file(argv[1], p);
    if (!g) { std::fprintf(stderr, "failed to read %s\n", argv[1]); return 1; }

    for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) {
        const char* key = gguf_get_key(g, i);
        gguf_type t = gguf_get_kv_type(g, i);
        std::printf("%-48s ", key);
        switch (t) {
            case GGUF_TYPE_STRING:  std::printf("str  %.60s\n", gguf_get_val_str(g, i)); break;
            case GGUF_TYPE_UINT32:  std::printf("u32  %u\n", gguf_get_val_u32(g, i)); break;
            case GGUF_TYPE_INT32:   std::printf("i32  %d\n", gguf_get_val_i32(g, i)); break;
            case GGUF_TYPE_FLOAT32: std::printf("f32  %g\n", gguf_get_val_f32(g, i)); break;
            case GGUF_TYPE_BOOL:    std::printf("bool %s\n", gguf_get_val_bool(g, i) ? "true" : "false"); break;
            case GGUF_TYPE_ARRAY:   std::printf("arr  %s[%zu]\n", gguf_type_name(gguf_get_arr_type(g, i)), gguf_get_arr_n(g, i)); break;
            default:                std::printf("%s\n", gguf_type_name(t)); break;
        }
    }
    gguf_free(g);
    return 0;
}
