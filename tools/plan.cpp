#include "planner.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

using namespace tierlm;
static double gib(double b) { return b / (1024.0 * 1024.0 * 1024.0); }
static double mib(double b) { return b / (1024.0 * 1024.0); }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s model.gguf [--vram GiB] [--ram GiB]\n", argv[0]); return 1; }
    Budget b;
    for (int i = 2; i + 1 < argc; i += 2) {
        uint64_t v = (uint64_t)(std::atof(argv[i + 1]) * 1024.0 * 1024.0 * 1024.0);
        if (!std::strcmp(argv[i], "--vram")) b.vram = v;
        else if (!std::strcmp(argv[i], "--ram")) b.ram = v;
    }
    ModelMap m;
    try { m = load_model_map(argv[1]); }
    catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
    Plan p = make_plan(m, b);

    std::printf("model %.2f GiB | budget: vram %.2f GiB, ram %.2f GiB\n\n", gib(m.total_bytes), gib(b.vram), gib(b.ram));

    std::map<Role, std::array<uint64_t, 3>> rt;
    for (const auto& u : p.units) rt[u.role][(int)u.tier] += u.bytes;
    std::printf("%-14s %9s %9s %9s\n", "role (GiB)", "vram", "ram", "ssd");
    for (const auto& [r, v] : rt) std::printf("%-14s %9.2f %9.2f %9.2f\n", role_name(r), gib(v[0]), gib(v[1]), gib(v[2]));
    std::printf("%-14s %9.2f %9.2f %9.2f\n", "total", gib(p.used[0]), gib(p.used[1]), gib(p.used[2]));

    if (m.n_experts > 0 && m.n_experts_used > 0) {
        std::map<int, std::array<int, 3>> cnt;
        std::array<double, 3> traffic{};
        for (const auto& u : p.units) {
            if (u.expert < 0) continue;
            cnt[u.layer][(int)u.tier]++;
            traffic[(int)u.tier] += (double)u.bytes * m.n_experts_used / m.n_experts;
        }
        int lo = INT_MAX, hi = 0;
        for (const auto& [l, c] : cnt) { lo = std::min(lo, c[0]); hi = std::max(hi, c[0]); }
        std::printf("\nexperts in vram per layer: %d to %d of %d\n", lo, hi, m.n_experts);
        std::printf("expert reads per token (uniform routing): vram %.1f MiB | ram %.1f MiB | ssd %.1f MiB\n",
                    mib(traffic[0]), mib(traffic[1]), mib(traffic[2]));
    }
    return 0;
}
