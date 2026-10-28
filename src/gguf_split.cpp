#include "gguf_split.h"
#include "gguf.h"
#include <cstdio>
#include <filesystem>
#include <regex>
#include <stdexcept>

namespace tierlm {
namespace fs = std::filesystem;

static const std::regex kPart(R"(^(.*)-(\d{5})-of-(\d{5})\.gguf$)");

bool gguf_is_later_part(const std::string& filename) {
    std::smatch m;
    return std::regex_match(filename, m, kPart) && std::stoi(m[2].str()) > 1;
}

SplitInfo gguf_split_info(const std::string& path) {
    SplitInfo s;
    s.paths.push_back(path);
    gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) throw std::runtime_error("failed to read GGUF: " + path);
    const int64_t id = gguf_find_key(g, "split.count");
    if (id >= 0) {
        switch (gguf_get_kv_type(g, id)) {
            case GGUF_TYPE_UINT16: s.count = gguf_get_val_u16(g, id); break;
            case GGUF_TYPE_UINT32: s.count = (int)gguf_get_val_u32(g, id); break;
            case GGUF_TYPE_INT32:  s.count = gguf_get_val_i32(g, id); break;
            default: break;
        }
    }
    gguf_free(g);
    if (s.count <= 1) { s.count = 1; return s; }

    const fs::path first = fs::u8path(path);
    const std::string name = first.filename().u8string();
    std::smatch m;
    if (!std::regex_match(name, m, kPart))
        throw std::runtime_error(name + " is part of a " + std::to_string(s.count) +
                                 "-part model, but its name doesn't follow the usual -00001-of-0000N.gguf pattern");
    s.paths.clear();
    for (int k = 1; k <= s.count; ++k) {
        char tail[48];
        std::snprintf(tail, sizeof(tail), "-%05d-of-%05d.gguf", k, s.count);
        const fs::path part = first.parent_path() / fs::u8path(m[1].str() + tail);
        std::error_code ec;
        if (s.missing.empty() && !fs::is_regular_file(part, ec)) s.missing = part.filename().u8string();
        s.paths.push_back(part.u8string());
    }
    return s;
}

}
