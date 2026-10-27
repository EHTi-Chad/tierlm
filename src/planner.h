#pragma once
#include "model_map.h"
#include <cstdint>
#include <string>
#include <vector>

namespace tierlm {

enum class Tier { VRAM = 0, RAM = 1, SSD = 2 };
const char* tier_name(Tier t);

struct Slice { uint64_t offset = 0, bytes = 0; std::string tensor; };

// Smallest thing the planner moves between tiers: a whole tensor,
// or one expert in one layer (its gate/up/down slices together).
struct Unit {
    std::string label;
    int layer = -1, expert = -1;
    Role role = Role::Other;
    uint64_t bytes = 0;
    std::vector<Slice> slices;
    Tier tier = Tier::SSD;
};

struct Budget { uint64_t vram = 0, ram = 0; };
struct Plan { std::vector<Unit> units; uint64_t used[3] = {0, 0, 0}; };

Plan make_plan(const ModelMap& m, const Budget& b);

}
