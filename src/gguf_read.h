// Reading a GGUF file's settings without trusting it. ggml's own getters (gguf_get_val_u32, gguf_get_arr_type,
// gguf_get_arr_str ...) stop the whole program with an assertion when a value has a different type than asked for,
// and gguf_get_arr_data hands back raw bytes that are easy to read past the end of when the element type isn't the
// one expected. A damaged or hostile model file must be refused with an error instead, so tierlm reads every
// setting through these: each checks the type, accepts any whole-number type where a whole number is wanted, and
// throws std::runtime_error naming the key when the value can't be used.
#pragma once
#include "gguf.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace tierlm::gg {

inline std::string key_name(const gguf_context* g, int64_t id) { return gguf_get_key(g, id); }

inline bool whole_type(gguf_type t) {
    switch (t) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_BOOL:
            return true;
        default: return false;
    }
}
inline bool number_type(gguf_type t) { return whole_type(t) || t == GGUF_TYPE_FLOAT32 || t == GGUF_TYPE_FLOAT64; }
inline bool is_array(const gguf_context* g, int64_t id) { return gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY; }

// Element i of raw data of type t (t a number type; the caller checked i against the length).
inline double element(gguf_type t, const void* d, size_t i) {
    switch (t) {
        case GGUF_TYPE_UINT8:   return ((const uint8_t*)d)[i];
        case GGUF_TYPE_INT8:    return ((const int8_t*)d)[i];
        case GGUF_TYPE_BOOL:    return ((const uint8_t*)d)[i] != 0;
        case GGUF_TYPE_UINT16:  return ((const uint16_t*)d)[i];
        case GGUF_TYPE_INT16:   return ((const int16_t*)d)[i];
        case GGUF_TYPE_UINT32:  return ((const uint32_t*)d)[i];
        case GGUF_TYPE_INT32:   return ((const int32_t*)d)[i];
        case GGUF_TYPE_UINT64:  return (double)((const uint64_t*)d)[i];
        case GGUF_TYPE_INT64:   return (double)((const int64_t*)d)[i];
        case GGUF_TYPE_FLOAT32: return ((const float*)d)[i];
        case GGUF_TYPE_FLOAT64: return ((const double*)d)[i];
        default: return 0;
    }
}
// A whole number that fits in int64 (the 64-bit types read exactly, not through a double).
inline int64_t element_int(gguf_type t, const void* d, size_t i) {
    if (t == GGUF_TYPE_UINT64) { const uint64_t v = ((const uint64_t*)d)[i]; return v > (uint64_t)INT64_MAX ? INT64_MAX : (int64_t)v; }
    if (t == GGUF_TYPE_INT64) return ((const int64_t*)d)[i];
    return (int64_t)element(t, d, i);
}

// One value (not a list).
inline int64_t int_val(const gguf_context* g, int64_t id) {
    const gguf_type t = gguf_get_kv_type(g, id);
    if (!whole_type(t)) throw std::runtime_error("setting " + key_name(g, id) + " should be a whole number");
    return element_int(t, gguf_get_val_data(g, id), 0);
}
inline double num_val(const gguf_context* g, int64_t id) {
    const gguf_type t = gguf_get_kv_type(g, id);
    if (!number_type(t)) throw std::runtime_error("setting " + key_name(g, id) + " should be a number");
    return element(t, gguf_get_val_data(g, id), 0);
}
inline bool bool_val(const gguf_context* g, int64_t id) { return int_val(g, id) != 0; }
inline std::string str_val(const gguf_context* g, int64_t id) {
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) throw std::runtime_error("setting " + key_name(g, id) + " should be text");
    return gguf_get_val_str(g, id);
}

// Lists. arr_n is 0 for a key that isn't a list.
inline size_t arr_n(const gguf_context* g, int64_t id) { return is_array(g, id) ? gguf_get_arr_n(g, id) : 0; }
inline bool num_array(const gguf_context* g, int64_t id) { return is_array(g, id) && number_type(gguf_get_arr_type(g, id)); }
inline bool str_array(const gguf_context* g, int64_t id) { return is_array(g, id) && gguf_get_arr_type(g, id) == GGUF_TYPE_STRING; }
inline void need_num_array(const gguf_context* g, int64_t id) {
    if (!num_array(g, id)) throw std::runtime_error("setting " + key_name(g, id) + " should be a list of numbers");
}
inline int64_t arr_int(const gguf_context* g, int64_t id, size_t i) {
    need_num_array(g, id);
    if (i >= gguf_get_arr_n(g, id)) throw std::runtime_error("setting " + key_name(g, id) + " is too short");
    const gguf_type t = gguf_get_arr_type(g, id);
    if (t == GGUF_TYPE_FLOAT32 || t == GGUF_TYPE_FLOAT64) throw std::runtime_error("setting " + key_name(g, id) + " should be whole numbers");
    return element_int(t, gguf_get_arr_data(g, id), i);
}
inline double arr_num(const gguf_context* g, int64_t id, size_t i) {
    need_num_array(g, id);
    if (i >= gguf_get_arr_n(g, id)) throw std::runtime_error("setting " + key_name(g, id) + " is too short");
    return element(gguf_get_arr_type(g, id), gguf_get_arr_data(g, id), i);
}
inline std::string arr_str(const gguf_context* g, int64_t id, size_t i) {
    if (!str_array(g, id)) throw std::runtime_error("setting " + key_name(g, id) + " should be a list of text");
    if (i >= gguf_get_arr_n(g, id)) throw std::runtime_error("setting " + key_name(g, id) + " is too short");
    return gguf_get_arr_str(g, id, i);
}

// Clamped to int (settings are sizes and counts; a value past INT_MAX is refused by the caller's range checks).
inline int clamp_int(int64_t v) { return v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : (int)v; }

}  // namespace tierlm::gg
