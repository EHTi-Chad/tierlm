// tierlm-devices: lists the compute devices this build can use, and explains why a GPU backend is missing.
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <sstream>
#include <string>
#ifndef TIERLM_VERSION
#define TIERLM_VERSION "dev"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

// Leftover VK_ICD_FILENAMES / VK_DRIVER_FILES force Vulkan to one driver file. After a driver update they
// often point at a file that no longer exists, and Vulkan then finds no GPU (or crashes while starting).
static void warn_vk_env() {
    for (const char* var : { "VK_ICD_FILENAMES", "VK_DRIVER_FILES" }) {
        const char* v = std::getenv(var);
        if (!v || !*v) continue;
#ifdef _WIN32
        const char sep = ';';
#else
        const char sep = ':';
#endif
        bool missing = false;
        std::stringstream ss(v);
        for (std::string p; std::getline(ss, p, sep);) {
            std::error_code ec;
            if (!p.empty() && !std::filesystem::exists(std::filesystem::u8path(p), ec)) missing = true;
        }
        std::printf("WARNING: %s is set%s:\n  %s\n", var, missing ? " to a driver file that does NOT exist" : "", v);
        std::printf("  This forces Vulkan to use one specific driver and can hide your GPU. To remove it:\n"
                    "    reg delete HKCU\\Environment /v %s /f\n  then sign out of Windows and back in.\n\n", var);
    }
    std::fflush(stdout);
}

int main() {
    std::printf("tierlm %s\n", TIERLM_VERSION);
    warn_vk_env();
#ifdef GGML_USE_VULKAN
    std::printf("Vulkan backend: built in\n");
    std::fflush(stdout);   // so this shows even if the driver crashes below
    try {
        int n = ggml_backend_vk_get_device_count();
        std::printf("Vulkan devices found: %d\n", n);
        for (int i = 0; i < n; ++i) {
            char desc[256] = {0};
            ggml_backend_vk_get_device_description(i, desc, sizeof(desc));
            std::printf("  Vulkan%d: %s\n", i, desc);
        }
    } catch (const std::exception& e) {
        std::printf("Vulkan failed to start: %s\n", e.what());
    } catch (...) {
        std::printf("Vulkan failed to start (unknown error)\n");
    }
#else
    std::printf("Vulkan backend: NOT built in (configure with -DTIERLM_VULKAN=ON)\n");
#endif
#ifdef GGML_USE_CUDA
    std::printf("CUDA backend: built in\n");
#endif

    std::printf("\nDevices registered with ggml:\n");
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(d, &free, &total);
        const char* type = "?";
        switch (ggml_backend_dev_type(d)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:  type = "CPU"; break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:  type = "GPU"; break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU: type = "integrated GPU"; break;
            default: break;
        }
        std::printf("  %-10s %-15s %s  (%.1f of %.1f GiB free)\n", ggml_backend_dev_name(d), type,
                    ggml_backend_dev_description(d), free / 1073741824.0, total / 1073741824.0);
    }
#ifdef NDEBUG
    std::printf("\nbuild type: Release\n");
#else
    std::printf("\nbuild type: Debug (slow! reconfigure with -DCMAKE_BUILD_TYPE=Release)\n");
#endif
    return 0;
}
