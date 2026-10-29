#include "backends.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#if defined(_WIN32) && defined(GGML_BACKEND_DL)
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>
#include <intrin.h>
#include <immintrin.h>
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace tierlm {

// ggml's log, minus one line per weight tensor when repacking (hundreds for a big model).
static void log_filter(ggml_log_level level, const char* text, void*) {
    if (std::strncmp(text, "repack: repack tensor", 21) == 0) return;
    if (level == GGML_LOG_LEVEL_DEBUG) return;
    std::fputs(text, stderr);
}

#if defined(_WIN32) && defined(GGML_BACKEND_DL)
// Windows release builds carry one CPU engine per x86 generation (ggml-cpu-<name>.dll). ggml picks one by loading
// every file and asking each for a score, so Windows (Smart App Control) checks, and may block, engines tierlm will
// never use. Instead tierlm reads the CPU's features itself and loads only the engine ggml would pick: the highest
// score among those the CPU can run, the first in name order on a tie. The scores are what each engine reports in
// an MSVC build (see ggml_backend_cpu_x86_score and ggml-cpu/CMakeLists.txt: MSVC defines only some feature macros).
// An engine is only chosen if the CPU really has every instruction set it was built with, which is stricter than
// ggml (on MSVC it doesn't check AVX512-VBMI), and the OS must save the AVX / AVX-512 registers.
namespace {
struct CpuFeats { bool sse42 = 0, avx = 0, f16c = 0, fma = 0, avx2 = 0, bmi2 = 0, avx_vnni = 0, avx512 = 0, vbmi = 0, vnni512 = 0, bf16 = 0, amx = 0; };
CpuFeats cpu_feats() {
    CpuFeats f;
    int r[4];
    __cpuid(r, 0);
    const int max_leaf = r[0];
    __cpuid(r, 1);
    const bool osxsave = r[2] & (1 << 27);
    const unsigned long long xcr0 = osxsave ? _xgetbv(0) : 0;
    const bool os_avx = (xcr0 & 0x6) == 0x6, os_avx512 = os_avx && (xcr0 & 0xE0) == 0xE0, os_amx = (xcr0 & 0x60000) == 0x60000;
    f.sse42 = r[2] & (1 << 20);
    f.avx = os_avx && (r[2] & (1 << 28));
    f.f16c = f.avx && (r[2] & (1 << 29));
    f.fma = f.avx && (r[2] & (1 << 12));
    if (max_leaf >= 7) {
        __cpuidex(r, 7, 0);
        const int b = r[1], c = r[2], d = r[3];
        f.avx2 = f.avx && (b & (1 << 5));
        f.bmi2 = b & (1 << 8);
        f.avx512 = os_avx512 && (b & (1 << 16)) && (b & (1 << 17)) && (b & (1 << 28)) && (b & (1 << 30)) && (b & (1u << 31));
        f.vbmi = f.avx512 && (c & (1 << 1));
        f.vnni512 = f.avx512 && (c & (1 << 11));
        f.amx = os_amx && (d & (1 << 24)) && (d & (1 << 25));
        __cpuidex(r, 7, 1);
        f.avx_vnni = f.avx2 && (r[0] & (1 << 4));
        f.bf16 = f.avx512 && (r[0] & (1 << 5));
    }
    return f;
}
// The CPU engine file to load, best first (the next ones are fallbacks if a file is missing or won't load).
std::vector<std::string> cpu_engine_order() {
    const CpuFeats c = cpu_feats();
    struct V { const char* name; int score; bool ok; };
    const bool hsw = c.fma && c.f16c && c.bmi2 && c.avx2;
    const V vs[] = {                                           // name order, as ggml finds them in the folder
        {"alderlake",      108,  hsw && c.avx_vnni},
        {"cannonlake",     137,  c.avx512 && c.bmi2 && c.vbmi},
        {"cascadelake",    1161, c.avx512 && c.bmi2 && c.vnni512},
        {"cooperlake",     1673, c.avx512 && c.bmi2 && c.vnni512 && c.bf16},
        {"haswell",        44,   hsw},
        {"icelake",        1161, c.avx512 && c.bmi2 && c.vbmi && c.vnni512},
        {"ivybridge",      17,   c.avx && c.f16c},
        {"piledriver",     17,   c.avx && c.f16c && c.fma},
        {"sandybridge",    17,   c.avx},
        {"sapphirerapids", 3721, c.avx512 && c.bmi2 && c.vbmi && c.vnni512 && c.bf16 && c.amx},
        {"skylakex",       137,  c.avx512 && c.bmi2},
        {"sse42",          5,    c.sse42},
        {"x64",            1,    true},
        {"zen4",           1673, c.avx512 && c.bmi2 && c.vbmi && c.vnni512 && c.bf16},
    };
    std::vector<V> ok;
    for (const V& v : vs) if (v.ok) ok.push_back(v);
    std::stable_sort(ok.begin(), ok.end(), [](const V& a, const V& b) { return a.score > b.score; });
    std::vector<std::string> out;
    for (const V& v : ok) out.push_back(v.name);
    return out;
}
std::filesystem::path exe_dir() {
    wchar_t buf[32768];
    const DWORD n = GetModuleFileNameW(nullptr, buf, 32768);
    return std::filesystem::path(std::wstring(buf, n)).parent_path();
}
// Loads only the engines tierlm uses; false if it couldn't (then ggml's own search runs instead).
bool load_backends_selectively() {
    const std::filesystem::path dir = exe_dir();
    const std::string d = dir.string();                          // ggml takes a narrow path: only for plain ones
    for (char ch : d) if ((unsigned char)ch >= 0x80) return false;
    bool cpu = false;
    for (const std::string& name : cpu_engine_order()) {
        const std::filesystem::path f = dir / ("ggml-cpu-" + name + ".dll");
        if (std::filesystem::exists(f) && ggml_backend_load(f.string().c_str())) { cpu = true; break; }
    }
    if (!cpu) {                                                  // a build with one plain CPU engine
        const std::filesystem::path f = dir / "ggml-cpu.dll";
        if (!std::filesystem::exists(f) || !ggml_backend_load(f.string().c_str())) return false;
    }
    for (const char* gpu : {"cuda", "vulkan", "hip", "sycl", "opencl", "rpc", "blas"}) {
        const std::filesystem::path f = dir / (std::string("ggml-") + gpu + ".dll");
        if (std::filesystem::exists(f)) ggml_backend_load(f.string().c_str());
    }
    return true;
}
} // namespace
#endif

void load_backends() {
    static std::once_flag once;
    std::call_once(once, [] {
        ggml_log_set(log_filter, nullptr);
#ifdef GGML_BACKEND_DL
#ifdef _WIN32
        if (!load_backends_selectively())
#endif
            ggml_backend_load_all();                            // the best CPU engine, then any GPU engines that load
#endif
    });
}

static ggml_backend_dev_t cpu_dev() {
    load_backends();
    ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!d) throw std::runtime_error("no CPU engine found: the ggml-cpu-*.dll files must be next to tierlm's programs");
    return d;
}

ggml_backend* cpu_backend_init() { return ggml_backend_dev_init(cpu_dev(), nullptr); }

ggml_backend_buffer_type* cpu_buffer_type() { return ggml_backend_dev_buffer_type(cpu_dev()); }

void set_n_threads(ggml_backend* backend, int n_threads) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) return;
    auto fn = (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (fn) fn(backend, n_threads);
}

std::string cpu_features() {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(cpu_dev());
    auto fn = (ggml_backend_get_features_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    std::string out;
    if (!fn) return out;
    for (const ggml_backend_feature* f = fn(reg); f && f->name; ++f) {
        // Instruction sets the engine was built for and uses ("1"), skipping version strings and the like.
        if (std::strcmp(f->value, "1") != 0) continue;
        if (!out.empty()) out += ' ';
        out += f->name;
    }
    return out;
}

bool add_rpc_server(const char* endpoints) {
    load_backends();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
    if (!reg) return false;
    using add_fn = ggml_backend_reg_t (*)(const char*);
    auto fn = (add_fn)ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_add_server");
    if (!fn) return false;
    const std::string all = endpoints;                     // host:port[,host:port...]: one stand-in GPU each
    for (size_t q = 0; q < all.size(); ) {
        size_t comma = all.find(',', q);
        if (comma == std::string::npos) comma = all.size();
        const std::string ep = all.substr(q, comma - q);
        q = comma + 1;
        if (ep.empty()) continue;
        ggml_backend_reg_t r = fn(ep.c_str());
        if (!r) return false;
        ggml_backend_register(r);
    }
    return true;
}

}
