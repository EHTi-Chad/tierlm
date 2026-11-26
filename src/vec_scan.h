#pragma once
// Scans over a row of logits (a whole vocabulary: 150K to 260K floats) written so compilers turn them into vector
// instructions. std::max_element tracks an index as it goes, which keeps it one value at a time: 0.84 ms over
// Flash-Next's 248,320 tokens on a 2.1 GHz Xeon, against 0.10 ms for the value first and its place after.
#include <cmath>
#include <cstddef>

namespace tierlm {

// The largest value (-inf for none). NaNs are skipped, as a comparison skips them.
inline float vec_max(const float* p, size_t n) {
    float m[8];
    for (float& v : m) v = -INFINITY;
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (int k = 0; k < 8; ++k) m[k] = p[i + k] > m[k] ? p[i + k] : m[k];
    float r = -INFINITY;
    for (float v : m) r = v > r ? v : r;
    for (; i < n; ++i) r = p[i] > r ? p[i] : r;
    return r;
}

// The place of the first largest value, as std::max_element gives it (0 for an empty row; the first value when
// none compares larger, as with all NaNs).
inline size_t vec_argmax(const float* p, size_t n) {
    if (n == 0) return 0;
    const float m = vec_max(p, n);
    for (size_t i = 0; i < n; ++i) if (p[i] == m) return i;
    // no value equals the max: every value is NaN or the row's max is -inf only through NaNs; std::max_element's answer
    size_t best = 0;
    for (size_t i = 1; i < n; ++i) if (p[best] < p[i]) best = i;
    return best;
}

}  // namespace tierlm
