#include "model_map.h"
#include "gguf_split.h"
#include "ggml.h"
#include "gguf.h"
#include <cstdlib>
#include <stdexcept>

namespace tierlm {

const char* role_name(Role r) {
    switch (r) {
        case Role::Embedding:    return "embedding";
        case Role::Output:       return "output";
        case Role::Attention:    return "attention";
        case Role::Norm:         return "norm";
        case Role::DenseFFN:     return "dense_ffn";
        case Role::Experts:      return "experts";
        case Role::SharedExpert: return "shared_expert";
        case Role::Router:       return "router";
        default:                 return "other";
    }
}

static Role classify(const std::string& n) {
    auto has = [&](const char* s) { return n.find(s) != std::string::npos; };
    if (has("token_embd"))                       return Role::Embedding;
    if (has("norm"))                             return Role::Norm;
    if (n.rfind("output", 0) == 0)               return Role::Output;
    if (has("ffn_gate_inp"))                     return Role::Router;
    if (has("_shexp"))                           return Role::SharedExpert;
    if (has("_exps"))                            return Role::Experts;
    if (has("attn_"))                            return Role::Attention;
    if (has("ffn_"))                             return Role::DenseFFN;
    return Role::Other;
}

static int get_u32(gguf_context* g, const std::string& key) {
    int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) return 0;
    return (int) gguf_get_val_u32(g, id);
}

ModelMap load_model_map(const std::string& path) {
    ggml_context* meta = nullptr;
    gguf_init_params params{ /*no_alloc=*/true, /*ctx=*/&meta };
    gguf_context* g = gguf_init_from_file(path.c_str(), params);
    if (!g) throw std::runtime_error("failed to read GGUF: " + path);

    ModelMap m;
    m.path = path;
    int64_t aid = gguf_find_key(g, "general.architecture");
    if (aid >= 0) m.arch = gguf_get_val_str(g, aid);
    m.n_layers       = get_u32(g, m.arch + ".block_count");
    m.n_experts      = get_u32(g, m.arch + ".expert_count");
    m.n_experts_used = get_u32(g, m.arch + ".expert_used_count");

    // Tensors can be spread over several files (a split model); part 1 holds the settings read above.
    const SplitInfo split = gguf_split_info(path);
    if (!split.missing.empty()) { gguf_free(g); ggml_free(meta); throw std::runtime_error("missing part: " + split.missing); }
    m.parts = split.count;
    for (int part = 0; part < split.count; ++part) {
        gguf_context* gp = g;
        ggml_context* mp = meta;
        if (part > 0) {
            mp = nullptr;
            gguf_init_params pp{ /*no_alloc=*/true, /*ctx=*/&mp };
            gp = gguf_init_from_file(split.paths[part].c_str(), pp);
            if (!gp) { gguf_free(g); ggml_free(meta); throw std::runtime_error("failed to read GGUF: " + split.paths[part]); }
        }
        const size_t data_off = gguf_get_data_offset(gp);
        const int64_t n = gguf_get_n_tensors(gp);
        for (int64_t i = 0; i < n; ++i) {
            TensorInfo t;
            t.name   = gguf_get_tensor_name(gp, i);
            t.type   = ggml_type_name(gguf_get_tensor_type(gp, i));
            t.bytes  = gguf_get_tensor_size(gp, i);
            t.offset = data_off + gguf_get_tensor_offset(gp, i);
            t.part   = part;
            t.role   = classify(t.name);
            if (t.name.rfind("blk.", 0) == 0) t.layer = std::atoi(t.name.c_str() + 4);
            if (ggml_tensor* tt = ggml_get_tensor(mp, t.name.c_str()))
                for (int d = 0; d < ggml_n_dims(tt); ++d) t.shape.push_back(tt->ne[d]);
            m.total_bytes += t.bytes;
            m.tensors.push_back(std::move(t));
        }
        if (part > 0) { gguf_free(gp); ggml_free(mp); }
    }
    gguf_free(g);
    ggml_free(meta);
    return m;
}

}
