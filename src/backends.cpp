#include "backends.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
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

namespace { bool g_cuda_none = false; }   // NVIDIA cards found, none the CUDA engine can run (see hide_unsupported_cuda)

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
        if (g_cuda_none && std::strcmp(gpu, "cuda") == 0) continue;   // no card it can run (see above)
        const std::filesystem::path f = dir / (std::string("ggml-") + gpu + ".dll");
        if (std::filesystem::exists(f)) ggml_backend_load(f.string().c_str());
    }
    return true;
}
} // namespace
#endif

// ---- NVIDIA cards the CUDA engine can't run ----
// A CUDA engine holds compiled code for the generations it was built for ("compute capabilities": 7.0 = Tesla V100,
// 7.5 = RTX 20, 8.6 = RTX 30, 8.9 = RTX 40, 12.0 = RTX 50). On a card it has no code for, the first GPU step fails
// and the program stops. So before the CUDA engine starts, tierlm asks NVIDIA's management library (NVML, installed
// with the driver; asking it doesn't start CUDA) which generation each card is, and lists only the covered ones in
// CUDA_VISIBLE_DEVICES. The others then appear only through Vulkan, and tierlm uses them there.
// Not done when CUDA_VISIBLE_DEVICES is already set (the user chose), or with TIERLM_CUDA_CHECK=0.
namespace {
std::vector<SkippedGpu> g_skipped;

struct Arch { int cc = 0; bool real = false, virt = false, exact = false; };
std::vector<Arch> built_archs() {
    std::vector<Arch> out;
#ifdef TIERLM_CUDA_ARCHS
    const std::string all = TIERLM_CUDA_ARCHS;                  // CMake's list, e.g. "70,75,86,89,120"
    for (size_t p = 0; p <= all.size(); ) {
        size_t q = all.find(',', p); if (q == std::string::npos) q = all.size();
        std::string e = all.substr(p, q - p); p = q + 1;
        if (e.empty()) continue;
        Arch a;
        size_t i = 0; while (i < e.size() && e[i] >= '0' && e[i] <= '9') ++i;
        if (i == 0) continue;
        a.cc = std::atoi(e.substr(0, i).c_str());
        std::string rest = e.substr(i);
        // ggml turns 12X into 12Xa (code for exactly that generation: its 4-bit tensor-core instructions)
        if (a.cc >= 120 && a.cc < 130 && (rest.empty() || rest[0] == '-')) rest = "a" + rest;
        if (!rest.empty() && (rest[0] == 'a' || rest[0] == 'f')) { a.exact = rest[0] == 'a'; rest = rest.substr(1); }
        if (rest.empty()) a.real = a.virt = true;               // no suffix: compiled code and portable code
        else if (rest == "-real") a.real = true;
        else if (rest == "-virtual") a.virt = true;
        out.push_back(a);
    }
#endif
    return out;
}
// Compiled code runs on its own generation and later minor versions of it (8.6 code on an 8.9 card); with the
// "a" ending only on exactly that one. Portable code (PTX) is compiled by the driver for any later card.
bool covered(int major, int minor, const std::vector<Arch>& archs) {
    const int cc = major * 10 + minor;
    for (const Arch& a : archs) {
        if (a.exact) { if (cc == a.cc) return true; continue; }
        if (a.real && a.cc / 10 == major && a.cc <= cc) return true;
        if (a.virt && a.cc <= cc) return true;
    }
    return false;
}

using nvml_fn0 = int (*)();
using nvml_count = int (*)(unsigned*);
using nvml_handle = int (*)(unsigned, void**);
using nvml_cc = int (*)(void*, int*, int*);
using nvml_str = int (*)(void*, char*, unsigned);
void* open_nvml(const char*& why) {
#ifdef _WIN32
    HMODULE h = LoadLibraryA("nvml.dll");                        // System32 with current drivers
    if (!h) {
        char pf[MAX_PATH] = {0};
        if (GetEnvironmentVariableA("ProgramFiles", pf, MAX_PATH))
            h = LoadLibraryA((std::string(pf) + "\\NVIDIA Corporation\\NVSMI\\nvml.dll").c_str());
    }
    why = "nvml.dll not found";
    return (void*)h;
#else
    void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    why = "libnvidia-ml.so.1 not found";
    return h;
#endif
}
void* sym(void* lib, const char* name) {
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}
void set_env(const char* k, const std::string& v) {
#ifdef _WIN32
    _putenv_s(k, v.c_str());
    SetEnvironmentVariableA(k, v.c_str());
#else
    setenv(k, v.c_str(), 1);
#endif
}
void hide_unsupported_cuda() {
    const std::vector<Arch> archs = built_archs();
    if (archs.empty()) return;                                   // not a CUDA build, or the list isn't known
    if (std::getenv("CUDA_VISIBLE_DEVICES")) return;             // the user chose the cards
    if (const char* v = std::getenv("TIERLM_CUDA_CHECK")) if (std::strcmp(v, "0") == 0) return;
    const char* why = nullptr;
    void* lib = open_nvml(why);
    if (!lib) return;                                            // no NVIDIA driver: CUDA won't start anyway
    auto init = (nvml_fn0)sym(lib, "nvmlInit_v2");
    auto shut = (nvml_fn0)sym(lib, "nvmlShutdown");
    auto count = (nvml_count)sym(lib, "nvmlDeviceGetCount_v2");
    auto handle = (nvml_handle)sym(lib, "nvmlDeviceGetHandleByIndex_v2");
    auto ccf = (nvml_cc)sym(lib, "nvmlDeviceGetCudaComputeCapability");
    auto uuid = (nvml_str)sym(lib, "nvmlDeviceGetUUID");
    auto namef = (nvml_str)sym(lib, "nvmlDeviceGetName");
    if (!init || !shut || !count || !handle || !ccf || !uuid || !namef || init() != 0) return;
    unsigned n = 0;
    std::vector<std::string> keep;
    std::vector<SkippedGpu> skip;
    bool ok = count(&n) == 0;
    for (unsigned i = 0; ok && i < n; ++i) {
        void* h = nullptr;
        int ma = 0, mi = 0;
        char id[96] = {0}, name[96] = {0};
        if (handle(i, &h) != 0 || ccf(h, &ma, &mi) != 0 || uuid(h, id, sizeof(id)) != 0) { ok = false; break; }
        namef(h, name, sizeof(name));
        if (covered(ma, mi, archs)) keep.push_back(id);
        else skip.push_back({name, ma, mi});
    }
    shut();
    if (!ok || skip.empty()) return;                             // unsure, or every card is covered: change nothing
    std::string list;
    for (const std::string& k : keep) { if (!list.empty()) list += ','; list += k; }
    set_env("CUDA_VISIBLE_DEVICES", list.empty() ? std::string("-1") : list);   // -1: no cards for CUDA
    g_skipped = skip;
    g_cuda_none = keep.empty();
    for (const SkippedGpu& g : skip)
        std::fprintf(stderr, "tierlm: %s (compute capability %d.%d) isn't covered by this build's CUDA engine (%s): it runs through Vulkan instead\n",
                     g.name.c_str(), g.major, g.minor, cuda_built_for().c_str());
}
} // namespace

const std::vector<SkippedGpu>& skipped_cuda_gpus() { load_backends(); return g_skipped; }

std::string cuda_built_for() {
    std::string out;
    for (const Arch& a : built_archs()) {
        const std::string s = std::to_string(a.cc / 10) + "." + std::to_string(a.cc % 10);
        if (out.find(s) != std::string::npos) continue;
        if (!out.empty()) out += ", ";
        out += s;
    }
    return out;
}

void load_backends() {
    static std::once_flag once;
    std::call_once(once, [] {
        ggml_log_set(log_filter, nullptr);
        hide_unsupported_cuda();                                // before the CUDA engine starts
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

// CPU threads. A model split between GPU and RAM runs dozens of small CPU steps per word (one per layer), with the
// GPU working in between. Built with OpenMP (the default), each step wakes OpenMP's threads. Built without it
// (TIERLM_OPENMP=OFF), each CPU backend gets its own ggml thread pool here, whose threads keep checking for work
// for a while after each step (TIERLM_CPU_POLL, 0-100, default 50: how long they spin before sleeping), so the next
// step starts without waking them. Without a pool ggml would start new threads for every step.
namespace {
struct CpuPool { ggml_threadpool* tp = nullptr; int n = 0; };
std::map<ggml_backend*, CpuPool>& cpu_pools() { static std::map<ggml_backend*, CpuPool> m; return m; }
std::mutex& cpu_pools_mu() { static std::mutex m; return m; }
bool cpu_has_openmp(ggml_backend_reg_t reg) {
    auto fn = (ggml_backend_get_features_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    if (!fn) return true;
    for (const ggml_backend_feature* f = fn(reg); f && f->name; ++f)
        if (std::strcmp(f->name, "OPENMP") == 0) return std::strcmp(f->value, "1") == 0;
    return false;
}
} // namespace

void set_n_threads(ggml_backend* backend, int n_threads) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) return;
    auto fn = (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (fn) fn(backend, n_threads);
    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) return;
    static const bool openmp = cpu_has_openmp(reg);
    if (openmp || n_threads < 1) return;
    using tp_new_t = ggml_threadpool* (*)(ggml_threadpool_params*);
    using tp_free_t = void (*)(ggml_threadpool*);
    using tp_set_t = void (*)(ggml_backend_t, ggml_threadpool*);
    auto tp_new = (tp_new_t)ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto tp_free = (tp_free_t)ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
    auto tp_set = (tp_set_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
    if (!tp_new || !tp_free || !tp_set) return;
    std::lock_guard<std::mutex> lk(cpu_pools_mu());
    CpuPool& p = cpu_pools()[backend];
    if (p.tp && p.n == n_threads) return;
    static const int poll = [] { const char* e = std::getenv("TIERLM_CPU_POLL"); return e ? std::clamp(std::atoi(e), 0, 100) : 50; }();
    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    tpp.poll = (uint32_t)poll;
    ggml_threadpool* tp = tp_new(&tpp);
    if (!tp) return;
    tp_set(backend, tp);
    if (p.tp) tp_free(p.tp);
    p.tp = tp; p.n = n_threads;
}

void forget_cpu_backend(ggml_backend* backend) {              // before the backend is freed: its pool goes too
    std::lock_guard<std::mutex> lk(cpu_pools_mu());
    auto it = cpu_pools().find(backend);
    if (it == cpu_pools().end()) return;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto tp_free = reg ? (void (*)(ggml_threadpool*))ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free") : nullptr;
    if (tp_free && it->second.tp) tp_free(it->second.tp);
    cpu_pools().erase(it);
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

std::recursive_mutex& device_mutex() { static std::recursive_mutex m; return m; }

}
