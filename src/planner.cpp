#include "planner.h"
#include <algorithm>
#include <map>

namespace tierlm {

const char* tier_name(Tier t) {
    switch (t) { case Tier::VRAM: return "vram"; case Tier::RAM: return "ram"; default: return "ssd"; }
}

// Lower = keep faster. Touched by every token ranks highest.
static int priority(Role r) {
    switch (r) {
        case Role::Norm: case Role::Router:            return 0;
        case Role::Attention:                          return 1;
        case Role::SharedExpert: case Role::DenseFFN:  return 2;
        case Role::Output:                             return 3;
        case Role::Embedding:                          return 3; // every token; never VRAM, guaranteed RAM
        default:                                       return 4; // experts, other
    }
}

Plan make_plan(const ModelMap& m, const Budget& b) {
    Plan p;
    std::map<std::pair<int, int>, Unit> experts;   // (layer, expert)
    for (const auto& t : m.tensors) {
        if (t.role == Role::Experts && m.n_experts > 0 && t.bytes % m.n_experts == 0) {
            const uint64_t per = t.bytes / m.n_experts;
            for (int e = 0; e < m.n_experts; ++e) {
                Unit& u = experts[{t.layer, e}];
                u.layer = t.layer; u.expert = e; u.role = Role::Experts;
                u.label = "blk." + std::to_string(t.layer) + " expert " + std::to_string(e);
                u.bytes += per;
                u.slices.push_back({t.offset + (uint64_t)e * per, per, t.name});
            }
        } else {
            Unit u; u.label = t.name; u.layer = t.layer; u.role = t.role; u.bytes = t.bytes;
            u.slices.push_back({t.offset, t.bytes, t.name});
            p.units.push_back(std::move(u));
        }
    }
    for (auto& kv : experts) p.units.push_back(std::move(kv.second));

    // No routing stats yet, so spread experts evenly: expert 0 of every layer, then expert 1, ...
    std::stable_sort(p.units.begin(), p.units.end(), [](const Unit& a, const Unit& c) {
        int pa = priority(a.role), pc = priority(c.role);
        if (pa != pc) return pa < pc;
        if (a.expert != c.expert) return a.expert < c.expert;
        return a.layer < c.layer;
    });

    for (auto& u : p.units) {
        if (u.role != Role::Embedding && p.used[0] + u.bytes <= b.vram) u.tier = Tier::VRAM;
        else if (p.used[1] + u.bytes <= b.ram)                          u.tier = Tier::RAM;
        else                                                            u.tier = Tier::SSD;
        p.used[(int)u.tier] += u.bytes;
    }
    return p;
}

}
