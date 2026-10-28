#include "backends.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace tierlm {

// ggml's log, minus one line per weight tensor when repacking (hundreds for a big model).
static void log_filter(ggml_log_level level, const char* text, void*) {
    if (std::strncmp(text, "repack: repack tensor", 21) == 0) return;
    if (level == GGML_LOG_LEVEL_DEBUG) return;
    std::fputs(text, stderr);
}

void load_backends() {
    static std::once_flag once;
    std::call_once(once, [] {
        ggml_log_set(log_filter, nullptr);
#ifdef GGML_BACKEND_DL
        ggml_backend_load_all();                                // the best CPU engine, then any GPU engines that load
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

bool add_rpc_server(const char* endpoint) {
    load_backends();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
    if (!reg) return false;
    using add_fn = ggml_backend_reg_t (*)(const char*);
    auto fn = (add_fn)ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_add_server");
    if (!fn) return false;
    ggml_backend_reg_t r = fn(endpoint);
    if (!r) return false;
    ggml_backend_register(r);
    return true;
}

}
