// tierlm-server: runs the engine behind a local web page with a live dashboard.
//
//   tierlm-server [model.gguf] [--port 8080] [--host 127.0.0.1] [-d auto|cpu|gpu|<device>] [-t threads] [-c context]
//                 [--web dir] [--models dir]
//   Without a model, the page opens a setup screen to choose the hardware and model.
//
// Endpoints
//   GET  /             the web page (web/index.html, re-read on every load so it can be edited live)
//   GET  /api/info     model, device and memory details ({"loaded": false} before a model is loaded)
//   GET  /api/devices  GPUs and CPU the engine can use, with memory
//   GET  /api/models   .gguf files in the models folder (and its subfolders), and whether each is supported
//   POST /api/folder   {"folder": "D:\\models", "recursive": true}  sets the models folder (saved)
//   POST /api/pick-folder  opens a folder picker on this computer (Windows; local requests only)
//   POST /api/load     {"model": "sub/file.gguf", "device": "auto|cpu|<device>", "ctx": 4096, "threads": 4,
//                       "flash_attn": true, "kv_type": "f16|q8_0|q4_0|f32", "batch": 256, "experts_used": 0, "seed": -1,
//                       "vram_limit_mb": 0, "split_mode": "auto|layers|experts"}
//   POST /api/unload   frees the model
//   POST /api/chat    {"messages":[{"role":"user","content":"..."}], "temperature":0.7, "top_p":0.9, "top_k":40,
//                     "min_p":0.05, "repeat_penalty":1.1, "repeat_last_n":64, "max_tokens":-1}
//                     streams Server-Sent Events: one "prompt" event, a "token" event per token
//                     (with the experts picked in every layer), then "done"
//   POST /api/reset   clears the KV cache
//   GET  /api/network  API and network settings, and the addresses other apps should use
//   POST /api/network  {"allow_network": true, "api_key": "..."}  (this computer only; saved)
//
// OpenAI-compatible API, for apps such as Open WebUI, Continue or anything built on the OpenAI SDK
// (base URL http://<this computer>:<port>/v1; see docs/API.md):
//   GET  /v1/models            the loaded model and every supported model in the models folder
//   POST /v1/chat/completions  chat, streamed or not; naming another model loads it first
#include "model.h"
#include "speculate.h"
#include "gguf_split.h"
#include "tokenizer.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "httplib.h"
#include "json.hpp"

#include <climits>
#include <algorithm>
#include <functional>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "backends.h"
#include <deque>
#ifdef _WIN32
#include <windows.h>
#include <shobjidl.h>
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#endif

#ifndef TIERLM_VERSION
#define TIERLM_VERSION "dev"
#endif

using namespace tierlm;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

namespace {

struct Args {
    std::string model, host = "127.0.0.1", device = "auto", web = "web", models = "models", settings = "tierlm-settings.json";
    std::string api_key;
    bool models_given = false, host_given = false, api_key_given = false;
    int port = 8080, threads = std::max(1, (int)std::thread::hardware_concurrency() / 2), ctx = 4096;
};

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string gguf_string(const std::string& path, const char* key) {
    gguf_init_params p{ true, nullptr };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) return {};
    int64_t id = gguf_find_key(g, key);
    std::string v = id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING ? gguf_get_val_str(g, id) : "";
    gguf_free(g);
    return v;
}

// Chat formats. Each model family wraps a conversation in its own markers; the GGUF's chat template (and
// failing that, its vocabulary) says which one. Control tokens are matched when encoding with parse_special.
enum class ChatFmt { Tulu, ChatML, Llama3, Mistral, Plain };

const char* chat_fmt_name(ChatFmt f) {
    switch (f) {
        case ChatFmt::Tulu: return "tulu"; case ChatFmt::ChatML: return "chatml";
        case ChatFmt::Llama3: return "llama3"; case ChatFmt::Mistral: return "mistral"; default: return "plain";
    }
}

ChatFmt detect_chat(const Tokenizer& tok) {
    const std::string& t = tok.chat_template();
    auto has = [&](const char* s) { return t.find(s) != std::string::npos; };
    if (has("<|im_start|>")) return ChatFmt::ChatML;
    if (has("<|start_header_id|>")) return ChatFmt::Llama3;
    if (has("<|user|>")) return ChatFmt::Tulu;
    if (has("[INST]")) return ChatFmt::Mistral;
    if (tok.find("<|im_start|>") >= 0) return ChatFmt::ChatML;
    if (tok.find("<|start_header_id|>") >= 0) return ChatFmt::Llama3;
    if (tok.find("<|user|>") >= 0) return ChatFmt::Tulu;
    if (tok.find("[INST]") >= 0 || tok.spm()) return ChatFmt::Mistral;
    return ChatFmt::Plain;
}

std::string format_chat(const json& messages, ChatFmt f, const Tokenizer& tok) {
    std::string s;
    switch (f) {
    case ChatFmt::Tulu: {                                 // OLMoE-Instruct / Tulu, and Zephyr (TinyLlama)
        const bool zephyr = tok.find("<|endoftext|>") < 0;      // Zephyr ends every turn with </s>
        const std::string end = zephyr ? "</s>" : "<|endoftext|>";
        s = zephyr ? (tok.bos() >= 0 ? "<s>" : "") : "<|endoftext|>";
        for (const auto& m : messages) {
            const std::string role = m.value("role", ""), c = m.value("content", "");
            if (role == "system")         s += "<|system|>\n" + c + (zephyr ? end : "") + "\n";
            else if (role == "user")      s += "<|user|>\n" + c + (zephyr ? end : "") + "\n";
            else if (role == "assistant") s += "<|assistant|>\n" + c + end + "\n";
        }
        return s + "<|assistant|>\n";
    }
    case ChatFmt::ChatML:                                 // Qwen, and many others
        for (const auto& m : messages)
            s += "<|im_start|>" + m.value("role", std::string("user")) + "\n" + m.value("content", std::string()) + "<|im_end|>\n";
        return s + "<|im_start|>assistant\n";
    case ChatFmt::Llama3:
        s = "<|begin_of_text|>";
        for (const auto& m : messages)
            s += "<|start_header_id|>" + m.value("role", std::string("user")) + "<|end_header_id|>\n\n" + m.value("content", std::string()) + "<|eot_id|>";
        return s + "<|start_header_id|>assistant<|end_header_id|>\n\n";
    case ChatFmt::Mistral: {                              // Llama 2 / Mistral: [INST] ... [/INST]
        const std::string bos = tok.bos() >= 0 ? "<s>" : "", eos = "</s>";
        std::string sys;
        s = bos;
        for (const auto& m : messages) {
            const std::string role = m.value("role", ""), c = m.value("content", "");
            if (role == "system") sys += c + "\n\n";
            else if (role == "user") { s += "[INST] " + sys + c + " [/INST]"; sys.clear(); }
            else if (role == "assistant") s += " " + c + eos;
        }
        return s;
    }
    default:
        for (const auto& m : messages) {
            const std::string role = m.value("role", "");
            s += (role == "assistant" ? "Assistant: " : role == "system" ? "System: " : "User: ") + m.value("content", std::string()) + "\n";
        }
        return s + "Assistant:";
    }
}

// Sampling settings, as in LM Studio / llama.cpp.
struct Sampling {
    float temperature = 0.7f, top_p = 0.9f, min_p = 0.05f, repeat_penalty = 1.1f;
    int top_k = 40, repeat_last_n = 64;
};

// Repetition penalty, then temperature with top-k, top-p (nucleus) and min-p filtering.
// temperature <= 0 means always take the most likely token (after the penalty).
int32_t sample(std::vector<float> lg, const Sampling& s, const std::vector<int32_t>& recent, std::mt19937& rng) {
    if (s.repeat_penalty > 0.0f && s.repeat_penalty != 1.0f) {
        // Make tokens used in the last repeat_last_n positions less likely (each token once).
        std::vector<int32_t> seen;
        const size_t n = std::min<size_t>(recent.size(), (size_t)std::max(0, s.repeat_last_n));
        for (size_t i = recent.size() - n; i < recent.size(); ++i) {
            const int32_t t = recent[i];
            if (t < 0 || (size_t)t >= lg.size() || std::find(seen.begin(), seen.end(), t) != seen.end()) continue;
            seen.push_back(t);
            lg[t] = lg[t] > 0 ? lg[t] / s.repeat_penalty : lg[t] * s.repeat_penalty;
        }
    }
    const int32_t best = (int32_t)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    if (s.temperature <= 0.0f) return best;
    const float mx = lg[best];
    std::vector<std::pair<float, int32_t>> p(lg.size());
    for (size_t i = 0; i < lg.size(); ++i) p[i] = {std::exp((lg[i] - mx) / s.temperature), (int32_t)i};
    size_t keep = s.top_k > 0 ? std::min<size_t>((size_t)s.top_k, p.size()) : std::min<size_t>(1024, p.size());
    std::partial_sort(p.begin(), p.begin() + keep, p.end(), [](auto& a, auto& b) { return a.first > b.first; });
    p.resize(keep);
    // min-p: drop tokens less than min_p times as likely as the best one (p[0].first is 1 after scaling).
    if (s.min_p > 0.0f) while (p.size() > 1 && p.back().first < s.min_p * p[0].first) p.pop_back();
    double total = 0;
    for (auto& x : p) total += x.first;
    double cum = 0;
    size_t n = 0;
    while (n < p.size() && cum < s.top_p * total) cum += p[n++].first;
    if (n == 0) n = 1, cum = p[0].first;
    std::uniform_real_distribution<double> u(0.0, cum);
    double r = u(rng);
    for (size_t i = 0; i < n; ++i) { r -= p[i].first; if (r <= 0) return p[i].second; }
    return p[n - 1].second;
}

// Length of the longest prefix of s that does not end inside a multi-byte UTF-8 character.
size_t utf8_complete(const std::string& s) {
    size_t i = s.size();
    for (int back = 0; i > 0 && back < 4; --i, ++back) {
        unsigned char c = (unsigned char)s[i - 1];
        if ((c & 0xC0) == 0x80) continue;                        // continuation byte: keep looking for the lead
        size_t need = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
        return s.size() - (i - 1) >= need ? s.size() : i - 1;
    }
    return s.size();
}

std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

namespace fs = std::filesystem;

// Settings kept between runs (models folder, per-model load settings), in a small JSON file.
json read_settings(const std::string& path) {
    json j = json::parse(read_file(path), nullptr, false);
    return j.is_object() ? j : json::object();
}
void write_settings(const std::string& path, const json& j) {
    const std::string tmp = path + ".tmp";
    { std::ofstream f(fs::u8path(tmp), std::ios::binary); if (!f) return; f << j.dump(2); }
    std::error_code ec;
    fs::rename(fs::u8path(tmp), fs::u8path(path), ec);
    if (ec) { fs::remove(fs::u8path(path), ec); fs::rename(fs::u8path(tmp), fs::u8path(path), ec); }
}

std::string env(const char* k) { const char* v = std::getenv(k); return v ? v : ""; }

// How often each expert of a model has been picked (n_layer x n_expert counts), kept between runs in
// tierlm-usage/<hash of the model's path>.bin next to the settings file. Loading a mixture-of-experts model that
// doesn't fit in VRAM uses it to keep the busiest experts in VRAM ("hot experts"), so tierlm learns each model's
// habits from use. Nothing but the counts is stored.
std::string usage_file(const std::string& settings_path, const std::string& model_path) {
    std::error_code ec;
    const std::string key = fs::weakly_canonical(fs::u8path(model_path), ec).generic_u8string();
    uint64_t hsh = 1469598103934665603ull;                      // FNV-1a
    for (unsigned char ch : key) { hsh ^= ch; hsh *= 1099511628211ull; }
    char name[32];
    std::snprintf(name, sizeof name, "%016llx.bin", (unsigned long long)hsh);
    return (fs::u8path(settings_path).parent_path() / "tierlm-usage" / name).u8string();
}
bool read_usage(const std::string& path, int L, int E, std::vector<float>& out) {
    std::ifstream f(fs::u8path(path), std::ios::binary);
    char magic[4]; int32_t l = 0, e = 0;
    if (!f.read(magic, 4) || std::memcmp(magic, "TLMU", 4) != 0 || !f.read((char*)&l, 4) || !f.read((char*)&e, 4) || l != L || e != E) return false;
    out.assign((size_t)L * E, 0.0f);
    return (bool)f.read((char*)out.data(), (std::streamsize)(out.size() * sizeof(float)));
}
void write_usage(const std::string& path, int L, int E, const std::vector<float>& u) {
    std::error_code ec;
    fs::create_directories(fs::u8path(path).parent_path(), ec);
    std::ofstream f(fs::u8path(path), std::ios::binary);
    if (!f) return;
    const int32_t l = L, e = E;
    f.write("TLMU", 4); f.write((const char*)&l, 4); f.write((const char*)&e, 4);
    f.write((const char*)u.data(), (std::streamsize)(u.size() * sizeof(float)));
}

// Folders where other apps keep GGUF models, offered as one-click choices if they exist.
std::vector<std::string> known_model_folders() {
    std::vector<std::string> out;
#ifdef _WIN32
    const std::string home = env("USERPROFILE");
#else
    const std::string home = env("HOME");
#endif
    if (home.empty()) return out;
    for (const char* sub : { ".lmstudio/models", ".cache/lm-studio/models" }) {
        std::error_code ec;
        fs::path p = fs::u8path(home) / fs::u8path(sub);
        if (fs::is_directory(p, ec)) out.push_back(p.make_preferred().u8string());
    }
    return out;
}

// Problems worth telling the user about before they wonder why their GPU is missing.
json hardware_warnings(bool have_gpu) {
    json w = json::array();
    for (const char* var : { "VK_ICD_FILENAMES", "VK_DRIVER_FILES" }) {
        const std::string v = env(var);
        if (v.empty()) continue;
        bool missing = false;
#ifdef _WIN32
        const char sep = ';';
#else
        const char sep = ':';
#endif
        std::stringstream ss(v);
        for (std::string part; std::getline(ss, part, sep);) {
            std::error_code ec;
            if (!part.empty() && !fs::exists(fs::u8path(part), ec)) missing = true;
        }
        w.push_back({{"code", "vk_env"}, {"variable", var}, {"value", v}, {"missing", missing},
                     {"message", std::string(var) + " is set" + (missing ? " and points at a driver file that no longer exists" : "") +
                                 ". This forces Vulkan to use one specific driver and is usually left over from an old driver install. "
                                 "It can hide your GPU." }});
    }
#ifdef GGML_USE_VULKAN
    if (!have_gpu)
        w.push_back({{"code", "no_gpu"}, {"message", "No GPU was found, so the CPU will be used. If this computer has a GPU, update its driver; "
                                                     "tierlm-devices.exe shows more detail."}});
#else
    (void)have_gpu;
#endif
    return w;
}

#ifdef _WIN32
// Opens the Windows folder picker on this computer. Runs on its own thread with its own COM apartment.
std::string pick_folder_windows(const std::string& start, bool& cancelled) {
    std::string result;
    cancelled = true;
    std::thread t([&] {
        if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE))) return;
        // A tiny always-on-top owner window, so the picker opens in front of the browser.
        HWND owner = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW, L"STATIC", L"tierlm", WS_POPUP, 0, 0, 0, 0,
                                     nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
        if (owner) { ShowWindow(owner, SW_SHOW); SetForegroundWindow(owner); }
        IFileOpenDialog* dlg = nullptr;
        if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dlg)))) {
            DWORD flags = 0;
            dlg->GetOptions(&flags);
            dlg->SetOptions(flags | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
            dlg->SetTitle(L"Choose the folder that holds your GGUF models");
            if (!start.empty()) {
                std::wstring w = fs::u8path(start).wstring();
                IShellItem* item = nullptr;
                if (SUCCEEDED(SHCreateItemFromParsingName(w.c_str(), nullptr, IID_PPV_ARGS(&item)))) { dlg->SetFolder(item); item->Release(); }
            }
            if (SUCCEEDED(dlg->Show(owner))) {
                IShellItem* item = nullptr;
                if (SUCCEEDED(dlg->GetResult(&item))) {
                    PWSTR p = nullptr;
                    if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &p))) {
                        result = fs::path(p).u8string();
                        cancelled = false;
                        CoTaskMemFree(p);
                    }
                    item->Release();
                }
            }
            dlg->Release();
        }
        if (owner) DestroyWindow(owner);
        CoUninitialize();
    });
    t.join();
    return result;
}
#endif


// The loaded model and everything that belongs to it.
struct Engine {
    std::unique_ptr<Speculator> spec;   // speculative decoding with the model's MTP draft block, when turned on
    int draft_max = 0;                  // most tokens the draft block guesses per check (0: 2 for MoE models, 3 for dense)
    std::unique_ptr<Model> model;
    Tokenizer tok;
    ChatFmt chat = ChatFmt::Plain;
    std::vector<int32_t> cached;      // the tokens currently in the KV cache
    std::string path, name;
    int threads = 1, seed = -1;
    double load_s = 0;
    std::mt19937 rng{std::random_device{}()};
    // Expert usage of the loaded model (n_layer x n_expert pick counts), learned from every prompt and reply.
    std::vector<float> usage;
    double hot_share = 0;                // share of the learned expert use that the VRAM experts cover
    std::string usage_path;
    bool usage_dirty = false;
    Clock::time_point usage_saved;
    int hot_E = 0, hot_L = 0;
    void add_usage(const std::vector<int32_t>& routing, int K) {
        if (usage.empty() || K <= 0) return;
        for (size_t i = 0; i < routing.size(); ++i) {
            const size_t l = (i / K) % hot_L;
            if (routing[i] >= 0 && routing[i] < hot_E) usage[l * hot_E + routing[i]] += 1.0f;
        }
        usage_dirty = true;
    }
    void save_usage(bool force = false) {
        if (!usage_dirty || usage_path.empty() || (!force && since(usage_saved) < 10.0)) return;
        // Older habits fade: once the counts get large, halve them, so recent use weighs more.
        double tot = 0; for (float v : usage) tot += v;
        if (tot > 4e6) for (float& v : usage) v *= 0.5f;
        write_usage(usage_path, hot_L, hot_E, usage);
        usage_dirty = false;
        usage_saved = Clock::now();
    }
};

// Header-only look at a GGUF file: what it is, how big, and whether tierlm can run it.
json describe_model(const std::string& path) {
    json j = {{"file", std::filesystem::u8path(path).filename().u8string()}, {"supported", false}};
    std::error_code ec;
    j["bytes"] = (uint64_t)std::filesystem::file_size(std::filesystem::u8path(path), ec);
    // A model split over several files: count every part's size, and make sure they're all there.
    SplitInfo split;
    try { split = gguf_split_info(path); } catch (const std::exception& e) { j["reason"] = e.what(); return j; }
    if (split.count > 1) {
        uint64_t total = 0;
        for (const auto& p : split.paths) { std::error_code e2; auto n = std::filesystem::file_size(std::filesystem::u8path(p), e2); if (!e2) total += n; }
        j["bytes"] = total;
        j["parts"] = split.count;
        if (!split.missing.empty()) {
            j["reason"] = "part of the model is missing: " + split.missing + " should be in the same folder";
            return j;
        }
    }
    gguf_init_params p{ true, nullptr };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) { j["reason"] = "not a readable GGUF file"; return j; }
    auto str = [&](const std::string& k) -> std::string {
        int64_t id = gguf_find_key(g, k.c_str());
        return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING ? gguf_get_val_str(g, id) : "";
    };
    auto u32 = [&](const std::string& k) -> int64_t {
        int64_t id = gguf_find_key(g, k.c_str());
        return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_UINT32 ? (int64_t)gguf_get_val_u32(g, id) : 0;
    };
    const std::string arch = str("general.architecture");
    j["arch"] = arch;
    j["name"] = str("general.name");
    const int64_t layers = u32(arch + ".block_count") - u32(arch + ".nextn_predict_layers"), embd = u32(arch + ".embedding_length");
    const int64_t heads = u32(arch + ".attention.head_count"), kv_heads = u32(arch + ".attention.head_count_kv");
    j["layers"] = layers;
    j["experts"] = u32(arch + ".expert_count");
    j["experts_used"] = u32(arch + ".expert_used_count");
    j["ctx_train"] = u32(arch + ".context_length");
    // KV cache cost per token of context, at f16: K and V for every layer.
    const int64_t head_dim = u32(arch + ".attention.key_length") ? u32(arch + ".attention.key_length") : (heads ? embd / heads : 0);
    // Only attention layers keep a KV cache; in hybrid models (qwen35, qwen35moe) that's one layer in every few.
    int64_t kv_layers = layers;
    // A multi-token-prediction (MTP) draft block, in the file or in a separate "mtp-" file next to it, makes
    // speculative decoding possible. A file that holds only a draft block is an add-on, not a model.
    const bool mtp_arch = arch == "qwen35" || arch == "qwen35moe" || arch == "qwen3next";
    const bool draft_only = u32(arch + ".nextn_predict_layers") > 0 && gguf_find_tensor(g, "blk.0.attn_norm.weight") < 0;
    if (draft_only) {
        j["kind"] = "draft";
        j["reason"] = "Draft add-on (MTP) that speeds up its model with speculative decoding; not a model on its own";
        gguf_free(g);
        return j;
    }
    if (mtp_arch && u32(arch + ".nextn_predict_layers") > 0) j["mtp"] = true;
    else if (mtp_arch) {
        const std::string comp = mtp_companion(path);
        if (!comp.empty()) { j["mtp"] = true; j["mtp_file"] = fs::u8path(comp).filename().u8string(); }
    }
    if (arch == "qwen35" || arch == "qwen35moe" || arch == "qwen3next") {
        const int64_t interval = u32(arch + ".full_attention_interval") ? u32(arch + ".full_attention_interval") : 4;
        kv_layers = layers / interval;
        j["hybrid"] = true;
    }
    if (layers && embd && heads) j["kv_bytes_per_token"] = 2 * kv_layers * (kv_heads ? kv_heads : heads) * head_dim * 2;
    const std::string tk = str("tokenizer.ggml.model");
    if (arch_supported(arch) && (tk == "gpt2" || tk == "llama")) j["supported"] = true;
    else if (arch_supported(arch)) j["reason"] = "the " + (tk.empty() ? std::string("unknown") : tk) + " tokenizer is not supported yet";
    else if (arch == "clip") {
        // A vision projector (mmproj file): the "eyes" of a vision model, not a model that runs on its own.
        j["kind"] = "mmproj";
        j["reason"] = "Vision add-on (mmproj) for a model in the same folder, not a standalone model";
    }
    else j["reason"] = "the " + (arch.empty() ? std::string("unknown") : arch) + " architecture is not supported yet";
    gguf_free(g);
    return j;
}

// ---- OpenAI-compatible API helpers ----

// OpenAI messages to plain {role, content} ones. Content may be a string, a list of parts
// ([{"type": "text", "text": ...}], images are skipped: no vision yet) or null. "developer" is OpenAI's newer
// name for "system"; tool results become user turns so the model still sees them.
json normalize_messages(const json& in) {
    json out = json::array();
    for (const auto& m : in) {
        if (!m.is_object()) continue;
        std::string role = m.value("role", std::string("user")), text;
        const json c = m.contains("content") ? m["content"] : json();
        if (c.is_string()) text = c.get<std::string>();
        else if (c.is_array())
            for (const auto& part : c) {
                if (part.is_string()) text += part.get<std::string>();
                else if (part.is_object() && part.value("type", std::string()) == "text") text += part.value("text", std::string());
            }
        if (role == "developer") role = "system";
        if (role == "tool" || role == "function") { role = "user"; text = "Tool result:\n" + text; }
        if (role != "system" && role != "user" && role != "assistant") role = "user";
        out.push_back({{"role", role}, {"content", text}});
    }
    return out;
}

void oai_error(httplib::Response& res, int status, const std::string& msg, const char* type = "invalid_request_error",
               const char* code = nullptr) {
    res.status = status;
    json e = {{"message", msg}, {"type", type}, {"param", nullptr}, {"code", code ? json(code) : json(nullptr)}};
    res.set_content(json({{"error", e}}).dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
}

// A request from this computer: a loopback address, and not passed on by a tunnel or proxy (Cloudflare Tunnel,
// ngrok and friends connect from localhost but add one of these headers), so it can't be someone outside.
bool is_local(const httplib::Request& req) {
    const std::string& a = req.remote_addr;
    const bool loop = a == "127.0.0.1" || a == "::1" || a == "::ffff:127.0.0.1";
    return loop && !req.has_header("X-Forwarded-For") && !req.has_header("Forwarded") &&
           !req.has_header("Cf-Connecting-Ip") && !req.has_header("X-Real-Ip");
}

// Same length and content, without returning early (so timing doesn't reveal how much of a key matched).
bool same_secret(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    unsigned char d = 0;
    for (size_t i = 0; i < a.size(); ++i) d |= (unsigned char)(a[i] ^ b[i]);
    return d == 0;
}

std::string random_key() {
    std::random_device rd;
    static const char* hex = "0123456789abcdef";
    std::string k = "tlm-";
    for (int i = 0; i < 40; ++i) k += hex[rd() % 16];
    return k;
}

// This computer's address on the local network, for showing other devices where to connect. Connecting a UDP
// socket sends nothing; it only asks the system which interface it would use.
std::string lan_address() {
    std::string ip;
#ifdef _WIN32
    WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    socket_t s = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (s == INVALID_SOCKET) return ip;
    sockaddr_in dst{}; dst.sin_family = AF_INET; dst.sin_port = htons(53);
    inet_pton(AF_INET, "10.255.255.255", &dst.sin_addr);
    if (::connect(s, (sockaddr*)&dst, sizeof(dst)) == 0) {
        sockaddr_in me{}; socklen_t n = sizeof(me);
        if (::getsockname(s, (sockaddr*)&me, &n) == 0) {
            char buf[INET_ADDRSTRLEN] = {};
            if (inet_ntop(AF_INET, &me.sin_addr, buf, sizeof(buf)) && std::string(buf) != "0.0.0.0") ip = buf;
        }
    }
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
    return ip;
}

// Everything tierlm and ggml print (the black window's text) is also kept for the page's console: stderr is
// redirected into a pipe, and a thread copies each line to the real stderr and into a ring of recent lines.
struct LogRing {
    std::mutex mu;
    std::deque<std::pair<long long, std::string>> lines;   // (line number, text)
    long long next = 0;
    void add(std::string t) {
        while (!t.empty() && (t.back() == '\r' || t.back() == '\n')) t.pop_back();
        std::lock_guard<std::mutex> lk(mu);
        lines.emplace_back(next++, std::move(t));
        while (lines.size() > 3000) lines.pop_front();
    }
};
LogRing g_log;

void capture_stderr() {
    int fds[2];
    std::fflush(stderr);
#ifdef _WIN32
    if (_pipe(fds, 1 << 16, _O_BINARY) != 0) return;
    const int orig = _dup(2);
    if (orig < 0 || _dup2(fds[1], 2) != 0) return;
    // Engines loaded later (the ggml DLLs, each with its own C runtime) pick up the pipe as their stderr too.
    SetStdHandle(STD_ERROR_HANDLE, (HANDLE)_get_osfhandle(fds[1]));
    auto rd = [](int fd, char* b, unsigned n) { return _read(fd, b, n); };
    auto wr = [](int fd, const char* b, unsigned n) { return _write(fd, b, n); };
#else
    if (pipe(fds) != 0) return;
    const int orig = dup(2);
    if (orig < 0 || dup2(fds[1], 2) < 0) return;
    auto rd = [](int fd, char* b, unsigned n) { return (int)read(fd, b, n); };
    auto wr = [](int fd, const char* b, unsigned n) { return (int)write(fd, b, n); };
#endif
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::thread([fd = fds[0], orig, rd, wr] {
        std::string part;
        char buf[4096];
        for (;;) {
            const int n = rd(fd, buf, sizeof buf);
            if (n <= 0) break;
            wr(orig, buf, (unsigned)n);
            part.append(buf, (size_t)n);
            size_t p;
            while ((p = part.find('\n')) != std::string::npos) { g_log.add(part.substr(0, p)); part.erase(0, p + 1); }
        }
    }).detach();
}

// Lines from ggml and the GPU drivers, shown in the console's Verbose view only.
bool verbose_line(const std::string& t) {
    static const char* pre[] = {"ggml_", "load_backend", "register_", "ggml-", "  Device", "found ", "Device ", "repack", "vk", "llama_"};
    for (const char* p : pre) if (t.compare(0, std::strlen(p), p) == 0) return true;
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    capture_stderr();
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (k == "--port") a.port = std::atoi(next().c_str());
        else if (k == "--host") { a.host = next(); a.host_given = true; }
        else if (k == "--api-key") { a.api_key = next(); a.api_key_given = true; }
        else if (k == "-d") a.device = next();
        else if (k == "-t") a.threads = std::atoi(next().c_str());
        else if (k == "-c") a.ctx = std::atoi(next().c_str());
        else if (k == "--web") a.web = next();
        else if (k == "--models") { a.models = next(); a.models_given = true; }
        else if (k == "--settings") a.settings = next();
        else if (k == "-h" || k == "--help") {
            std::fprintf(stderr, "usage: %s [model.gguf] [--port 8080] [--host 127.0.0.1] [--api-key KEY] [-d auto|cpu|gpu|<device>] [-t threads] [-c context] [--web dir] [--models dir] [--settings file]\n"
                                 "Without a model, the web page opens a setup screen to choose the hardware and model.\n", argv[0]);
            return 0;
        }
        else if (a.model.empty()) a.model = k;
    }

    load_backends();                     // in release builds: the best CPU engine for this PC, and the GPU engines
    // Testing hook: TIERLM_RPC=host:port adds a tierlm-rpc-server as a stand-in GPU.
    if (const char* ep = std::getenv("TIERLM_RPC")) add_rpc_server(ep);
    Engine eng;
    eng.threads = a.threads;
    std::timed_mutex mu;                 // one load or generation at a time (a new request waits up to 3 s for the last one to finish)
    std::atomic<bool> loading{false};    // a model is being loaded (for the page's progress bar)
    std::atomic<float> load_progress{0.0f};

    // The models folder: --models if given, else the one saved last time, else "models".
    json settings = read_settings(a.settings);
    bool recursive = settings.value("recursive", true);
    if (!a.models_given && settings.contains("models_folder") && settings["models_folder"].is_string()) {
        std::error_code ec;
        const std::string saved = settings["models_folder"];
        if (fs::is_directory(fs::u8path(saved), ec)) a.models = saved;
    }
    std::fprintf(stderr, "tierlm %s\nmodels folder: %s\n", TIERLM_VERSION, a.models.c_str());

    // hot: keep the model's busiest experts in VRAM, from what tierlm learned about it before (MoE models, experts in RAM).
    auto load = [&](const std::string& path, const LoadOptions& opt, int threads, int seed, bool hot = true) {
        json info = describe_model(path);
        if (!info.value("supported", false)) throw std::runtime_error(info.value("reason", std::string("unsupported model")));
        eng.save_usage(true);            // what the previous model learned
        eng.spec.reset();
        eng.model.reset();               // free the old model (and its GPU memory) first
        eng.cached.clear();
        auto t0 = Clock::now();
        eng.tok = Tokenizer::from_gguf(path);
        LoadOptions o = opt;
        const int uL = info.value("layers", 0), uE = info.value("experts", 0);
        eng.usage.clear(); eng.usage_path.clear(); eng.usage_dirty = false; eng.hot_L = uL; eng.hot_E = uE;
        if (uE > 0 && uL > 0) {
            eng.usage_path = usage_file(a.settings, path);
            if (!read_usage(eng.usage_path, uL, uE, eng.usage)) eng.usage.assign((size_t)uL * uE, 0.0f);
            double tot = 0; for (float v : eng.usage) tot += v;
            if (hot && tot > 0) o.expert_usage = eng.usage;
            eng.usage_saved = Clock::now();
        }
        o.progress = [&](float f) { load_progress = f; };
        load_progress = 0.0f;
        loading = true;
        struct Done { std::atomic<bool>& b; ~Done() { b = false; } } done{loading};
        eng.model = Model::load(path, o);
        // Each guess an MoE model checks pulls in its own experts, so shorter guesses pay off there
        // (Qwen3.6-35B-A3B, RTX 3070 laptop: 35 tok/s with 1 or 2, 28 with 3); dense models do best with 3.
        if (eng.draft_max <= 0) eng.draft_max = eng.model->hparams().is_moe() ? 2 : 3;
        if (opt.mtp && eng.model->has_mtp()) eng.spec = std::make_unique<Speculator>(*eng.model, eng.draft_max);
        eng.chat = detect_chat(eng.tok);
        eng.path = path;
        eng.name = gguf_string(path, "general.name");
        eng.threads = threads > 0 ? threads : a.threads;
        eng.seed = seed;
        if (seed >= 0) eng.rng.seed((uint32_t)seed);
        eng.load_s = since(t0);
        std::fprintf(stderr, "loaded %s (%s, %s chat format) on %s in %.1fs\n", path.c_str(), eng.model->arch().c_str(),
                     chat_fmt_name(eng.chat), eng.model->device_name().c_str(), eng.load_s);
        const Placement& pl = eng.model->placement();
        if (pl.hot_experts > 0) {
            int split = 0; for (size_t l = 0; l < pl.hot.size(); ++l) split += !pl.hot[l].empty() && !pl.ffn_vram[l];
            // How much of the past expert work the VRAM experts would have done: what the CPU is spared.
            double all = 0, in_vram = 0;
            for (int l = 0; l < uL && (size_t)l < pl.hot.size(); ++l) {
                for (int e = 0; e < uE; ++e) all += eng.usage[(size_t)l * uE + e];
                if (l < (int)pl.ffn_vram.size() && pl.ffn_vram[l])
                    for (int e = 0; e < uE; ++e) in_vram += eng.usage[(size_t)l * uE + e];
                else
                    for (int e : pl.hot[l]) in_vram += eng.usage[(size_t)l * uE + e];
            }
            eng.hot_share = all > 0 ? in_vram / all : 0;
            std::fprintf(stderr, "hot experts: %d of %d experts in VRAM, chosen from earlier use (%d layers split between VRAM and RAM); "
                         "they did %.0f%% of the expert work so far\n", pl.hot_experts, uL * uE, split, 100 * eng.hot_share);
        }
    };
    // Load settings from a /api/load body (or the settings saved for a model).
    auto load_options = [&](const json& body) {
        LoadOptions o;
        o.device = body.value("device", std::string("auto"));
        o.max_ctx = body.value("ctx", a.ctx);
        if (body.contains("flash_attn")) o.flash_attn = body.value("flash_attn", true) ? 1 : 0;
        if (body.contains("kv_type")) o.kv_type = body.value("kv_type", std::string("f16"));   // f16, f32, q8_0, q4_0
        o.batch = body.value("batch", 0);
        o.experts_used = body.value("experts_used", 0);
        o.vram_budget_mb = body.value("vram_limit_mb", 0);
        o.split_mode = body.value("split_mode", std::string("auto"));
        o.mtp = body.value("speculative", false);      // load the MTP draft block for speculative decoding
        o.multi_gpu = body.value("multi_gpu", true);
        return o;
    };
    if (!a.model.empty()) {
        LoadOptions o; o.max_ctx = a.ctx; o.device = a.device;
        try { load(a.model, o, a.threads, -1); }
        catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
    }

    httplib::Server svr;

    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        std::string page = read_file(a.web + "/index.html");
        if (page.empty()) {
            res.status = 404;
            res.set_content("web/index.html not found. Run the server from the tierlm folder, or pass --web <folder>.", "text/plain");
            return;
        }
        res.set_content(page, "text/html; charset=utf-8");
    });

    json last_info = {{"loaded", false}, {"version", TIERLM_VERSION}};
    svr.Get("/api/info", [&](const httplib::Request&, httplib::Response& res) {
        // While a reply is streaming or a model is loading, answer from the last snapshot instead of waiting.
        std::unique_lock<std::timed_mutex> lk(mu, std::try_to_lock);
        if (!lk.owns_lock()) { json j = last_info; j["busy"] = true; res.set_content(dump(j), "application/json"); return; }
        if (!eng.model) { last_info = {{"loaded", false}, {"version", TIERLM_VERSION}}; res.set_content(dump(last_info), "application/json"); return; }
        const auto& hp = eng.model->hparams();
        size_t free = 0, total = 0;
        eng.model->device_memory(free, total);
        json j = {
            {"loaded", true}, {"path", eng.path},
            {"name", eng.name.empty() ? fs::u8path(eng.path).stem().u8string() : eng.name}, {"arch", eng.model->arch()},
            {"chat_format", chat_fmt_name(eng.chat)}, {"moe", hp.is_moe()},
            {"layers", hp.n_layer}, {"experts", hp.n_expert}, {"experts_used", hp.n_expert_used},
            {"vocab", hp.n_vocab}, {"ctx", eng.model->max_ctx()}, {"n_past", eng.model->n_past()},
            {"device", eng.model->device_name()}, {"on_gpu", eng.model->on_gpu()}, {"threads", eng.threads},
            {"device_type", eng.model->integrated_gpu() ? "igpu" : eng.model->on_gpu() ? "gpu" : "cpu"},
            {"weights_bytes", eng.model->weights_bytes()}, {"kv_bytes", eng.model->kv_bytes()},
            {"device_free", free}, {"device_total", total},
            {"kv_type", eng.model->kv_type()}, {"flash_attn", eng.model->flash_attn()}, {"load_s", eng.load_s},
            {"batch", eng.model->batch()}, {"experts_default", eng.model->experts_default()}, {"seed", eng.seed},
            {"vram_weights", eng.model->placement().gpu_weights}, {"ram_weights", eng.model->placement().cpu_weights},
            {"kv_on_gpu", eng.model->placement().kv_on_gpu}, {"attn_gpu", eng.model->placement().attn_gpu},
            {"ffn_gpu", eng.model->placement().ffn_gpu}, {"kv_gpu_layers", eng.model->placement().kv_gpu_layers},
            {"split_mode", eng.model->placement().mode},
            {"gpus", [&] { json g = json::array(); for (const auto& x : eng.model->placement().gpus)
                               g.push_back({{"name", x.name}, {"weights", x.weights}, {"kv", x.kv}, {"layers", x.layers}}); return g; }()},
            {"speculative", eng.spec != nullptr}, {"draft_max", eng.draft_max}, {"has_draft_block", hp.n_nextn > 0},
            {"mtp_loaded", eng.model->has_mtp()}, {"ffn_vram", eng.model->placement().ffn_vram},
            {"hot_experts", eng.model->placement().hot_experts}, {"hot_share", eng.hot_share}, {"hot", eng.model->placement().hot},
            {"learned_tokens", [&] { double t = 0; for (float v : eng.usage) t += v; return hp.n_layer * hp.n_expert_used > 0 ? (long long)(t / (hp.n_layer * hp.n_expert_used)) : 0LL; }()},
            {"version", TIERLM_VERSION},
        };
        last_info = j;
        res.set_content(dump(j), "application/json");
    });

    // Hardware the engine can use, with memory, for the setup screen.
    svr.Get("/api/devices", [&](const httplib::Request&, httplib::Response& res) {
        json list = json::array();
        bool have_gpu = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(d, &free, &total);
            const char* type = "other";
            switch (ggml_backend_dev_type(d)) {
                case GGML_BACKEND_DEVICE_TYPE_CPU:  type = "cpu"; break;
                case GGML_BACKEND_DEVICE_TYPE_GPU:  type = "gpu"; break;
                case GGML_BACKEND_DEVICE_TYPE_IGPU: type = "igpu"; break;
                default: break;
            }
            if (std::strcmp(type, "gpu") == 0 || std::strcmp(type, "igpu") == 0) have_gpu = true;
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d);
            list.push_back({{"id", ggml_backend_dev_name(d)}, {"description", ggml_backend_dev_description(d)},
                            {"type", type}, {"backend", reg ? ggml_backend_reg_name(reg) : ""},
                            {"free", free}, {"total", total}});
        }
        // One NVIDIA card shows up as both CUDA0 and Vulkan0: mark the Vulkan entry as the same GPU, so the
        // page shows the card once (CUDA is the faster route on NVIDIA).
        for (auto& d : list) {
            if (d["backend"] != "Vulkan") continue;
            for (const auto& c : list)
                if (c["backend"] == "CUDA" && c["description"] == d["description"]) d["same_gpu_as"] = c["id"];
        }
        json j = {{"devices", list}, {"threads", a.threads}, {"hw_threads", (int)std::thread::hardware_concurrency()},
                  {"cpu_features", cpu_features()},
                  {"warnings", hardware_warnings(have_gpu)}, {"version", TIERLM_VERSION}};
        res.set_content(dump(j), "application/json");
    });

    // Models in the models folder (and, if enabled, its subfolders, as LM Studio stores them), read from
    // their headers only (fast: no weights are loaded). "id" is the path relative to the folder.
    // Every .gguf in the models folder (subfolders too if enabled), described from its header.
    auto scan_models = [&](bool& exists) {
        json list = json::array();
        std::error_code ec;
        const fs::path base = fs::u8path(a.models);
        auto add = [&](const fs::path& p) {
            json m = describe_model(p.u8string());
            m["id"] = p.lexically_relative(base).generic_u8string();
            std::string sub = p.parent_path().lexically_relative(base).generic_u8string();
            m["folder"] = sub == "." ? "" : sub;
            list.push_back(m);
        };
        exists = fs::is_directory(base, ec);
        if (exists && recursive) {
            auto it = fs::recursive_directory_iterator(base, fs::directory_options::skip_permission_denied, ec);
            for (; !ec && it != fs::recursive_directory_iterator() && list.size() < 500; it.increment(ec)) {
                if (it.depth() > 4) { it.disable_recursion_pending(); continue; }
                std::error_code e2;
                if (it->is_regular_file(e2) && it->path().extension() == ".gguf" && !gguf_is_later_part(it->path().filename().u8string())) add(it->path());
            }
        } else if (exists) {
            for (const auto& f : fs::directory_iterator(base, ec))
                if (f.is_regular_file() && f.path().extension() == ".gguf" && !gguf_is_later_part(f.path().filename().u8string())) add(f.path());
        }
        return list;
    };
    svr.Get("/api/models", [&](const httplib::Request&, httplib::Response& res) {
        bool exists = false;
        json list = scan_models(exists);
        std::error_code ec;
        const fs::path base = fs::u8path(a.models);
        json suggest = json::array();
        for (const auto& f : known_model_folders()) {
            std::error_code e2;
            if (!fs::equivalent(fs::u8path(f), base, e2)) suggest.push_back({{"name", "LM Studio models"}, {"folder", f}});
        }
        json per_model = settings.value("models", json::object());
        json j = {{"folder", fs::absolute(base, ec).make_preferred().u8string()}, {"exists", exists}, {"recursive", recursive},
                  {"models", list}, {"default_ctx", a.ctx}, {"current", eng.path}, {"suggestions", suggest},
                  {"saved", per_model}, {"last_model", settings.value("last_model", std::string())},
#ifdef _WIN32
                  {"can_pick", true},
#else
                  {"can_pick", false},
#endif
        };
        res.set_content(dump(j), "application/json");
    });

    // Change the models folder. Saved for next time.
    svr.Post("/api/folder", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) { res.status = 400; res.set_content(dump({{"error", "bad request"}}), "application/json"); return; }
        std::lock_guard<std::timed_mutex> lk(mu);
        if (body.contains("recursive")) recursive = body.value("recursive", true);
        if (body.contains("folder")) {
            std::string f = body.value("folder", std::string());
            while (f.size() > 1 && (f.back() == ' ' || f.back() == '"')) f.pop_back();
            while (!f.empty() && (f.front() == ' ' || f.front() == '"')) f.erase(0, 1);
            std::error_code ec;
            if (f.empty() || !fs::is_directory(fs::u8path(f), ec)) {
                res.status = 400;
                res.set_content(dump({{"error", "folder not found: " + f}}), "application/json");
                return;
            }
            a.models = fs::absolute(fs::u8path(f), ec).make_preferred().u8string();
            settings["models_folder"] = a.models;
        }
        settings["recursive"] = recursive;
        write_settings(a.settings, settings);
        res.set_content(dump({{"ok", true}, {"folder", a.models}, {"recursive", recursive}}), "application/json");
    });

    // Native folder picker. It opens on the computer running tierlm, so only answer local requests.
    svr.Post("/api/pick-folder", [&](const httplib::Request& req, httplib::Response& res) {
#ifdef _WIN32
        if (req.remote_addr != "127.0.0.1" && req.remote_addr != "::1") {
            res.status = 403;
            res.set_content(dump({{"error", "the folder picker only works from the computer running tierlm"}}), "application/json");
            return;
        }
        bool cancelled = true;
        std::string f = pick_folder_windows(a.models, cancelled);
        res.set_content(dump({{"cancelled", cancelled}, {"folder", f}}), "application/json");
#else
        (void)req;
        res.status = 501;
        res.set_content(dump({{"error", "type the folder path instead (the picker is Windows only)"}}), "application/json");
#endif
    });

    // A model id from the page or the API, as a file inside the models folder (empty if it isn't one: only files
    // inside the models folder can be loaded).
    auto model_file = [&](const std::string& id) -> std::filesystem::path {
        std::error_code ec;
        const auto base = std::filesystem::weakly_canonical(std::filesystem::u8path(a.models), ec);
        const auto file = std::filesystem::weakly_canonical(std::filesystem::u8path(a.models) / std::filesystem::u8path(id), ec);
        auto rel = file.lexically_relative(base);
        if (ec || rel.empty() || rel.begin()->string() == ".." || !std::filesystem::is_regular_file(file, ec)) return {};
        return file;
    };

    svr.Post("/api/load", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("model")) {
            res.status = 400;
            res.set_content("{\"error\":\"expected {\\\"model\\\": \\\"...\\\"}\"}", "application/json");
            return;
        }
        if (!mu.try_lock_for(std::chrono::seconds(3))) {
            res.status = 409;
            res.set_content("{\"error\":\"busy: wait for the current reply or load to finish\"}", "application/json");
            return;
        }
        std::lock_guard<std::timed_mutex> lk(mu, std::adopt_lock);
        const auto file = model_file(body.value("model", std::string()));
        if (file.empty()) {
            res.status = 400;
            res.set_content(dump({{"error", "model not found in the models folder"}}), "application/json");
            return;
        }
        try {
            const LoadOptions o = load_options(body);
            eng.draft_max = std::min(8, std::max(0, body.value("draft_max", 0)));
            const int threads = body.value("threads", a.threads), seed = body.value("seed", -1);
            load(file.u8string(), o, threads, seed, body.value("hot_experts", false));
            // Remember these settings for this model, like LM Studio does.
            const std::string id = body.value("model", std::string());
            json saved = body;
            saved.erase("model");
            settings["models"][id] = saved;
            settings["last_model"] = id;
            write_settings(a.settings, settings);
            res.set_content(dump({{"ok", true}, {"device", eng.model->device_name()}, {"load_s", eng.load_s}}), "application/json");
        } catch (const std::exception& e) {
            eng.spec.reset();
            eng.model.reset();
            eng.path.clear();
            res.status = 500;
            res.set_content(dump({{"error", e.what()}}), "application/json");
        }
    });

    svr.Post("/api/unload", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::timed_mutex> lk(mu);
        eng.save_usage(true);
        eng.spec.reset();
        eng.model.reset();
        eng.cached.clear();
        eng.path.clear();
        res.set_content("{\"ok\":true}", "application/json");
    });

    svr.Post("/api/reset", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::timed_mutex> lk(mu);
        if (eng.model) eng.model->reset();
        eng.cached.clear();
        res.set_content("{\"ok\":true}", "application/json");
    });

    // One reply, shared by the web page and the OpenAI-compatible API. The caller holds `mu`.
    //   on_prompt  once, after the conversation has been read
    //   on_token   for each piece of new text (with the experts each layer picked); return false to stop
    // `stops` are strings that end the reply (OpenAI's "stop"); text that might be the start of one is held back.
    struct Gen { size_t n_prompt = 0, reused = 0, n_gen = 0, drafted = 0, accepted = 0, checks = 0; double prompt_s = 0, gen_s = 0;
                 std::string stop = "length", text; };
    auto generate = [&](const json& messages, const Sampling& sp, int max_tokens, const std::vector<std::string>& stops,
                        const std::function<void(const json&)>& on_prompt,
                        const std::function<bool(const std::string&, double, const std::vector<int32_t>&)>& on_token) {
        Gen r;
        if (max_tokens <= 0) max_tokens = INT_MAX;    // no limit: until the model finishes or the context is full
        if (eng.seed >= 0) eng.rng.seed((uint32_t)eng.seed);    // same seed, same reply
        auto& model = eng.model; auto& tok = eng.tok; auto& cached = eng.cached;
        const auto& hp = model->hparams();
        const int L = hp.n_layer, E = hp.n_expert, K = hp.n_expert_used;

        std::vector<int32_t> toks = tok.encode(format_chat(messages, eng.chat, tok), true);
        if ((int)toks.size() + 1 >= model->max_ctx()) throw std::runtime_error("conversation is longer than the context window; start a new chat");

        // Reuse whatever part of the KV cache still matches this conversation.
        size_t common = 0;
        while (common < cached.size() && common < toks.size() && cached[common] == toks[common]) ++common;
        if (common == toks.size()) common = toks.size() - 1;      // must evaluate at least one token
        model->truncate((int)common);
        common = (size_t)model->n_past();     // hybrid models start over rather than rewind
        cached.resize(common);

        std::vector<int32_t> suffix(toks.begin() + common, toks.end()), routing;
        auto p0 = Clock::now();
        std::vector<float> lg;
        if (eng.spec) { lg = eng.spec->prompt(toks, eng.threads, &routing); eng.spec->reset_stats(); }   // also feeds the draft block
        else lg = model->eval(suffix, eng.threads, &routing);
        r.prompt_s = since(p0);
        std::fprintf(stderr, "chat: %zu prompt tokens, %zu reused from the cache, %zu read at %.0f tok/s\n", toks.size(), (size_t)common,
                     toks.size() - common, (toks.size() - common) / std::max(r.prompt_s, 1e-9));
        r.n_prompt = toks.size();
        r.reused = common;
        cached = toks;

        std::vector<int> counts((size_t)L * E, 0);                // experts used while reading the prompt
        if (K > 0) for (size_t i = 0; i < routing.size(); ++i) counts[((i / K) % L) * E + routing[i]]++;
        eng.add_usage(routing, K);
        if (on_prompt)
            on_prompt({{"n_prompt", toks.size()}, {"n_new", suffix.size()}, {"reused", common}, {"prompt_s", r.prompt_s},
                       {"prompt_tps", suffix.size() / std::max(r.prompt_s, 1e-9)}, {"counts", counts}, {"n_past", model->n_past()}});

        std::vector<int32_t> out;
        size_t shown = 0;
        auto g0 = Clock::now();
        auto emit = [&](const std::string& visible, size_t upto) {
            if (upto <= shown) return true;
            const std::string delta = visible.substr(shown, upto - shown);
            shown = upto;
            r.text += delta;
            return on_token ? on_token(delta, out.size() / std::max(since(g0), 1e-9), routing) : true;
        };
        std::string visible;
        auto is_end = [&](int32_t t) { return t == tok.eos() || t == tok.eot() || tok.is_control(t); };
        // Recent tokens for the repetition penalty (the cache, then `extra`), leaving out control tokens such as the
        // end-of-reply marker: penalising those would stop the model from finishing.
        auto recent_with = [&](const std::vector<int32_t>& extra) {
            std::vector<int32_t> rc;
            for (size_t j = extra.size(); j > 0 && (int)rc.size() < sp.repeat_last_n; --j) if (!is_end(extra[j - 1])) rc.push_back(extra[j - 1]);
            for (size_t j = cached.size(); j > 0 && (int)rc.size() < sp.repeat_last_n; --j) if (!is_end(cached[j - 1])) rc.push_back(cached[j - 1]);
            return rc;
        };
        // Adds a chosen token to the reply and streams its text. False when the reply must end here.
        auto add = [&](int32_t id) {
            out.push_back(id);
            const std::string text = tok.decode(out);
            visible = text.substr(0, utf8_complete(text));
            // A stop string: end the reply just before it.
            size_t cut = std::string::npos;
            for (const auto& st : stops)
                if (!st.empty()) { size_t p = visible.find(st, shown > st.size() ? shown - st.size() : 0); if (p < cut) cut = p; }
            if (cut != std::string::npos) { emit(visible, std::max(cut, shown)); r.stop = "stop"; return false; }
            // Hold back text that could be the start of a stop string.
            size_t hold = 0;
            for (const auto& st : stops)
                for (size_t k = st.empty() ? 0 : std::min(st.size() - 1, visible.size()); k > hold; --k)
                    if (visible.compare(visible.size() - k, k, st, 0, k) == 0) { hold = k; break; }
            if (!emit(visible, visible.size() - hold)) { r.stop = "abort"; return false; }
            return true;
        };
        if (!eng.spec) {
            for (;;) {
                if ((int)out.size() >= max_tokens) { r.stop = "length"; break; }
                const int32_t id = sample(lg, sp, recent_with({}), eng.rng);
                if (is_end(id)) { r.stop = "end"; break; }
                if (model->n_past() + 1 >= model->max_ctx()) { r.stop = "context"; break; }
                lg = model->eval({id}, eng.threads, &routing);            // also tells us which experts it used
                eng.add_usage(routing, K);
                cached.push_back(id);
                if (!add(id)) break;
            }
        } else {
            // Speculative decoding: each check yields one or more tokens, all chosen by the model itself.
            std::vector<int32_t> kept;                                    // experts picked for x and the accepted guesses
            const size_t per_tok = (size_t)L * std::max(K, 0);
            auto routing_of = [&](size_t j) {                             // the dashboard shows one token's picks at a time
                if (per_tok == 0 || kept.size() < (j + 1) * per_tok) routing.clear();
                else routing.assign(kept.begin() + j * per_tok, kept.begin() + (j + 1) * per_tok);
            };
            int32_t cur = -1;                                             // the token being checked
            auto pick = [&](const float* row, const std::vector<int32_t>& acc) {
                std::vector<int32_t> extra;
                if (cur >= 0) extra.push_back(cur);
                extra.insert(extra.end(), acc.begin(), acc.end());
                return sample(std::vector<float>(row, row + hp.n_vocab), sp, recent_with(extra), eng.rng);
            };
            int32_t x = pick(lg.data(), {});
            for (;;) {
                if ((int)out.size() >= max_tokens) { r.stop = "length"; break; }
                if (is_end(x)) { r.stop = "end"; break; }
                if (model->n_past() + 1 >= model->max_ctx()) { r.stop = "context"; break; }
                cur = x;
                const std::vector<int32_t> prod = eng.spec->step(x, eng.threads, pick, is_end, K > 0 ? &kept : nullptr);
                eng.add_usage(kept, K);
                cur = -1;
                cached.push_back(x);                                      // x and the accepted guesses are now cached
                cached.insert(cached.end(), prod.begin(), prod.end() - 1);
                routing_of(0);
                bool go = add(x);
                for (size_t j = 0; go && j + 1 < prod.size(); ++j) {
                    if ((int)out.size() >= max_tokens) { r.stop = "length"; go = false; break; }
                    routing_of(j + 1);
                    go = add(prod[j]);
                }
                if (!go) break;
                x = prod.back();
            }
            const SpecStats& st = eng.spec->stats();
            r.drafted = st.drafted; r.accepted = st.accepted; r.checks = st.steps;
        }
        if (r.stop != "stop" && r.stop != "abort") emit(visible, visible.size());   // release held-back text
        r.n_gen = out.size();
        r.gen_s = since(g0);
        eng.save_usage();
        if (r.checks > 0)
            std::fprintf(stderr, "reply: %zu tokens at %.1f tok/s (%s); speculative: %.1f tokens per check, %zu of %zu guesses kept\n", r.n_gen,
                         r.n_gen / std::max(r.gen_s, 1e-9), r.stop.c_str(), (double)r.n_gen / r.checks, r.accepted, r.drafted);
        else
            std::fprintf(stderr, "reply: %zu tokens at %.1f tok/s (%s)\n", r.n_gen, r.n_gen / std::max(r.gen_s, 1e-9), r.stop.c_str());
        return r;
    };

    svr.Post("/api/chat", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content("{\"error\":\"expected {\\\"messages\\\": [...]}\"}", "application/json");
            return;
        }
        if (!mu.try_lock_for(std::chrono::seconds(3))) {
            res.status = 409;
            res.set_content("{\"error\":\"busy: a reply is being generated or a model is loading\"}", "application/json");
            return;
        }
        if (!eng.model) {
            mu.unlock();
            res.status = 409;
            res.set_content("{\"error\":\"no model loaded: choose one on the setup screen\"}", "application/json");
            return;
        }
        auto lock = std::make_shared<std::unique_lock<std::timed_mutex>>(mu, std::adopt_lock);   // released when the stream ends
        res.set_header("Cache-Control", "no-cache");

        res.set_chunked_content_provider("text/event-stream", [&, body, lock](size_t, httplib::DataSink& sink) {
            auto send = [&](const json& j) { std::string s = "data: " + dump(j) + "\n\n"; return sink.write(s.data(), s.size()); };
            try {
                Sampling sp;
                sp.temperature = body.value("temperature", sp.temperature);
                sp.top_p = body.value("top_p", sp.top_p);
                sp.top_k = body.value("top_k", sp.top_k);
                sp.min_p = body.value("min_p", sp.min_p);
                sp.repeat_penalty = body.value("repeat_penalty", sp.repeat_penalty);
                sp.repeat_last_n = body.value("repeat_last_n", sp.repeat_last_n);
                // Longest reply in tokens; 0 or less (the default) means no limit: the reply ends when the model
                // finishes or the context window is full. Thinking models (Qwen3.x) need thousands of tokens.
                const int max_tokens = body.value("max_tokens", -1);
                const int L = eng.model->hparams().n_layer, K = eng.model->hparams().n_expert_used;
                Gen g = generate(body["messages"], sp, max_tokens, {},
                    [&](const json& p) { json j = p; j["type"] = "prompt"; send(j); },
                    [&](const std::string& delta, double tps, const std::vector<int32_t>& routing) {
                        json picks = json::array();
                        if (K > 0 && routing.size() >= (size_t)L * K)
                            for (int l = 0; l < L; ++l)
                                picks.push_back(std::vector<int32_t>(routing.begin() + (size_t)l * K, routing.begin() + (size_t)(l + 1) * K));
                        return send({{"type", "token"}, {"text", delta}, {"tps", tps}, {"n_past", eng.model->n_past()}, {"routing", picks}});
                    });
                json done = {{"type", "done"}, {"n_gen", g.n_gen}, {"gen_s", g.gen_s},
                             {"gen_tps", g.n_gen / std::max(g.gen_s, 1e-9)}, {"stop", g.stop}, {"n_past", eng.model->n_past()}};
                if (g.checks > 0) done["spec"] = {{"checks", g.checks}, {"drafted", g.drafted}, {"accepted", g.accepted}};
                send(done);
            } catch (const std::exception& e) {
                send({{"type", "error"}, {"message", e.what()}});
            }
            sink.done();
            return true;
        });
    });


    // ---------------- Speed measurement (auto-tune and benchmark) ----------------
    // The text every measurement generates after; long enough to be a realistic prompt.
    const std::string text =
        "Explain, step by step, how a computer's memory hierarchy works: registers, caches, main memory and "
        "storage. For each level, describe how fast and how big it typically is, why the trade-off exists, "
        "and give an everyday analogy. Then explain why programs that reuse nearby data run faster.";
    auto is_end = [&](int32_t t) { return t == eng.tok.eos() || t == eng.tok.eot() || eng.tok.is_control(t); };
    auto best_token = [&](const float* row) {
        int32_t b = -1;
        for (int32_t i = 0; i < eng.model->hparams().n_vocab; ++i)
            if (!is_end(i) && (b < 0 || row[i] > row[b])) b = i;
        return b;
    };
    // Generation speed (tokens/second) with the loaded model: n tokens after the prompt, timed after a
    // short warm-up. draft = 0: no speculation; otherwise guesses per check.
    const int warm = 4;
    auto measure = [&](int n, int draft) -> double {
        eng.model->reset();
        eng.cached.clear();
        const std::vector<int32_t> toks = eng.tok.encode(text);
        n = std::max(4, std::min(n, eng.model->max_ctx() - (int)toks.size() - warm - draft - 8));
        std::vector<float> lg;
        std::unique_ptr<Speculator> sp;
        if (draft > 0) { sp = std::make_unique<Speculator>(*eng.model, draft); lg = sp->prompt(toks, eng.threads); }
        else lg = eng.model->eval(toks, eng.threads);
        int32_t x = best_token(lg.data());
        auto greedy = [&](const float* row, const std::vector<int32_t>&) { return best_token(row); };
        int made = 0, from = -1;
        Clock::time_point tw;
        for (;;) {
            if (from < 0 && made >= warm) { from = made; tw = Clock::now(); }
            if (from >= 0 && made - from >= n) break;
            if (sp) {
                const std::vector<int32_t> prod = sp->step(x, eng.threads, greedy, [](int32_t) { return false; });
                made += (int)prod.size();
                x = prod.back();
            } else {
                lg = eng.model->eval({x}, eng.threads);
                x = best_token(lg.data());
                ++made;
            }
        }
        const double tps = (made - from) / std::max(since(tw), 1e-9);
        eng.model->reset();
        eng.cached.clear();
        return tps;
    };
    // Prompt reading speed (tokens/second): about n tokens of text read in one go, as a long prompt would be.
    auto measure_prompt = [&](int n) -> double {
        eng.model->reset();
        eng.cached.clear();
        std::vector<int32_t> toks;
        const std::vector<int32_t> one = eng.tok.encode(text + " ");
        n = std::min(n, eng.model->max_ctx() - 8);
        while ((int)toks.size() < n) toks.insert(toks.end(), one.begin(), one.end());
        toks.resize(n);
        const auto t = Clock::now();
        eng.model->eval(toks, eng.threads);
        const double tps = n / std::max(since(t), 1e-9);
        eng.model->reset();
        return tps;
    };

    // ---------------- Auto-tune ----------------
    // Finds the fastest settings for a model on this PC by measuring them: how to split it when it doesn't fit in
    // VRAM, how many CPU threads to use, and whether (and how far ahead) to guess with speculative decoding. The
    // settings that change the answers (context, context memory precision, sampling) are left as the body gives
    // them. Streams {"type":"status"|"result"|"done"|"error"} events; the winner is saved for the model and stays
    // loaded.
    //
    // Each measurement generates a fixed text greedily (end-of-reply tokens skipped, so every run is the same
    // length) after a warm-up, and times only the generation. Searched one setting at a time, in the order that
    // matters most: split mode (a reload each), then threads, then guesses per check.
    svr.Post("/api/autotune", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("model")) {
            res.status = 400;
            res.set_content("{\"error\":\"expected {\\\"model\\\": \\\"...\\\"}\"}", "application/json");
            return;
        }
        const auto file = model_file(body.value("model", std::string()));
        if (file.empty()) {
            res.status = 400;
            res.set_content(dump({{"error", "model not found in the models folder"}}), "application/json");
            return;
        }
        if (!mu.try_lock_for(std::chrono::seconds(3))) {
            res.status = 409;
            res.set_content("{\"error\":\"busy: wait for the current reply or load to finish\"}", "application/json");
            return;
        }
        auto lock = std::make_shared<std::unique_lock<std::timed_mutex>>(mu, std::adopt_lock);
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream", [&, body, file, lock](size_t, httplib::DataSink& sink) {
            bool alive = true;
            auto send = [&](const json& j) {
                std::string s = "data: " + dump(j) + "\n\n";
                if (alive && !sink.write(s.data(), s.size())) alive = false;
                return alive;
            };
            auto status = [&](const std::string& t) { std::fprintf(stderr, "autotune: %s\n", t.c_str()); return send({{"type", "status"}, {"text", t}}); };
            try {
                const json info = describe_model(file.u8string());
                if (!info.value("supported", false)) throw std::runtime_error(info.value("reason", std::string("unsupported model")));
                const bool has_mtp = info.value("mtp", false);
                const int hw = std::max(1, (int)std::thread::hardware_concurrency());
                const int t0 = std::max(1, body.value("threads", a.threads));
                const int seed = body.value("seed", -1);
                LoadOptions base = load_options(body);
                base.mtp = has_mtp;                          // the draft layer stays loaded so guesses can be tried

                json results = json::array();
                auto record = [&](const std::string& label, json setting, double tps) {
                    setting["label"] = label;
                    setting["tps"] = std::round(tps * 10) / 10;
                    results.push_back(setting);
                    std::fprintf(stderr, "autotune: %s: %.1f tok/s\n", label.c_str(), tps);
                    return send({{"type", "result"}, {"label", label}, {"tps", setting["tps"]}});
                };
                int n_gen = 32;                               // set from the first measurement: about 5 s per run
                auto size_runs = [&](double tps) { n_gen = std::max(12, std::min(96, (int)std::lround(tps * 5))); };

                // ---- 1. split mode (when the model doesn't fit in VRAM) ----
                std::string mode = "auto";
                std::string loaded_mode;
                // Split modes: "layers", "experts" (whole expert layers in VRAM), "hot" (experts in RAM, with the
                // busiest experts in VRAM, from what tierlm has learned about this model).
                auto load_mode = [&](const std::string& m) {
                    LoadOptions o = base;
                    o.split_mode = m == "hot" ? "experts" : m;
                    eng.draft_max = 0;
                    load(file.u8string(), o, t0, seed, m == "hot");
                    loaded_mode = m;
                };
                auto learned = [&] { double t = 0; for (float v : eng.usage) t += v; return t; };
                // Learns which experts this model uses from a mix of everyday text, code and maths, when nothing has
                // been learned yet (afterwards, every chat keeps teaching it).
                auto calibrate = [&] {
                    static const char* texts[] = {
                        "Write a friendly email to a colleague asking to move tomorrow's meeting to Thursday afternoon.",
                        "def merge_sort(items):\n    \"\"\"Sort a list with merge sort and explain each step.\"\"\"\n",
                        "Explain why the sky is blue, then work out 17 * 23 and 144 / 12, showing each step.",
                        "Summarize the main causes of the First World War in a short paragraph for a history class.",
                        "Translate into French and Spanish: The library opens at nine and closes at six on weekdays."};
                    const int L_ = eng.model->hparams().n_layer, K_ = eng.model->hparams().n_expert_used;
                    for (const char* t : texts) {
                        eng.model->reset();
                        std::vector<int32_t> routing;
                        std::vector<float> lg = eng.model->eval(eng.tok.encode(t), eng.threads, &routing);
                        eng.add_usage(routing, K_);
                        for (int i = 0; i < 48 && eng.model->n_past() + 2 < eng.model->max_ctx(); ++i) {
                            const int32_t x = best_token(lg.data());
                            lg = eng.model->eval({x}, eng.threads, &routing);
                            eng.add_usage(routing, K_);
                        }
                    }
                    (void)L_;
                    eng.model->reset();
                    eng.cached.clear();
                    eng.save_usage(true);
                };
                const char* mode_label[] = {"whole layers", "experts in RAM", "experts in RAM, busiest experts in VRAM"};
                if (!status("Loading the model")) throw std::runtime_error("cancelled");
                load_mode("layers");
                const Placement& pl = eng.model->placement();
                const bool split = eng.model->on_gpu() && pl.split();
                const bool cpu_work = split || !eng.model->on_gpu();
                {
                    double t = measure(8, 0);                // a first, short run to size the others
                    size_runs(t);
                }
                double best = 0;
                if (split) {
                    const std::string modes[] = {"layers", "experts", "hot"};
                    const int n_modes = eng.model->hparams().is_moe() ? 3 : 2;
                    for (int i = 0; i < n_modes; ++i) {
                        if (i == 2 && learned() <= 0) {
                            if (!status("Learning which experts this model uses")) throw std::runtime_error("cancelled");
                            calibrate();
                        }
                        if (loaded_mode != modes[i]) {
                            if (!status(std::string("Loading with ") + mode_label[i])) throw std::runtime_error("cancelled");
                            load_mode(modes[i]);
                        }
                        if (i == 2 && eng.model->placement().hot_experts == 0) {   // nothing to gain: no VRAM left for experts
                            loaded_mode = "experts";
                            continue;
                        }
                        if (!status(std::string("Measuring ") + mode_label[i])) throw std::runtime_error("cancelled");
                        const double t = measure(n_gen, 0);
                        if (!record(std::string("Split: ") + mode_label[i], {{"split_mode", modes[i]}}, t)) throw std::runtime_error("cancelled");
                        if (t > best) { best = t; mode = modes[i]; }
                    }
                    if (loaded_mode != mode) {
                        if (!status("Loading the faster split again")) throw std::runtime_error("cancelled");
                        load_mode(mode);
                    }
                    size_runs(best);
                } else {
                    if (!status("Measuring")) throw std::runtime_error("cancelled");
                    best = measure(n_gen, 0);
                    record(std::string(eng.model->on_gpu() ? "All in VRAM" : "CPU only") + (cpu_work ? ", " + std::to_string(t0) + " CPU threads" : ""),
                           json::object(), best);
                    mode = "auto";
                }

                // ---- 2. CPU threads (when the CPU does part of the work) ----
                int threads = t0;
                if (cpu_work) {
                    std::vector<int> cands = {t0, std::max(1, t0 * 3 / 4), std::min(hw, (t0 * 3 + 1) / 2)};
                    std::sort(cands.begin(), cands.end());
                    cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
                    for (int c : cands) {
                        if (c == t0) continue;               // measured above
                        eng.threads = c;
                        if (!status("Measuring " + std::to_string(c) + " CPU threads")) throw std::runtime_error("cancelled");
                        const double t = measure(n_gen, 0);
                        if (!record(std::to_string(c) + " CPU threads", {{"threads", c}}, t)) throw std::runtime_error("cancelled");
                        if (t > best * 1.02) { best = t; threads = c; }   // a clear win only: the default is a safe choice
                    }
                    eng.threads = threads;
                }

                // ---- 3. speculative decoding (when the model has a draft layer) ----
                int draft = 0;
                if (has_mtp && eng.model->has_mtp()) {
                    const double plain = best;
                    for (int d = 1; d <= 4; ++d) {
                        if (!status("Measuring speculative decoding, " + std::to_string(d) + (d == 1 ? " guess" : " guesses") + " per check"))
                            throw std::runtime_error("cancelled");
                        const double t = measure(n_gen, d);
                        if (!record("Speculative decoding, " + std::to_string(d) + (d == 1 ? " guess" : " guesses"), {{"draft_max", d}}, t))
                            throw std::runtime_error("cancelled");
                        if (t > best) { best = t; draft = d; }
                    }
                    if (draft > 0 && best < plain * 1.03) draft = 0;    // not worth it for a sliver
                    if (draft == 0) best = plain;
                }

                // ---- the winner: keep it loaded and remember it ----
                eng.draft_max = draft > 0 ? draft : 0;
                eng.spec.reset();
                if (draft > 0) eng.spec = std::make_unique<Speculator>(*eng.model, draft);
                else eng.draft_max = eng.model->hparams().is_moe() ? 2 : 3;
                eng.model->reset();
                eng.cached.clear();
                const std::string id = body.value("model", std::string());
                json saved = body;
                saved.erase("model");
                saved["split_mode"] = split ? (mode == "hot" ? "experts" : mode) : body.value("split_mode", std::string("auto"));
                saved["hot_experts"] = mode == "hot";                          // on only where it measured fastest
                saved["threads"] = threads;
                saved["speculative"] = draft > 0;
                if (draft > 0) saved["draft_max"] = draft;
                saved["settings_v"] = 2;
                saved["tuned"] = {{"tps", std::round(best * 10) / 10}, {"results", results}};
                settings["models"][id] = saved;
                settings["last_model"] = id;
                write_settings(a.settings, settings);
                send({{"type", "done"}, {"tps", std::round(best * 10) / 10}, {"settings", saved}, {"results", results}});
            } catch (const std::exception& e) {
                send({{"type", "error"}, {"message", e.what()}});
            }
            sink.done();
            return true;
        });
    });


    // ---------------- Benchmark: the loaded model's speed with its current settings ----------------
    svr.Post("/api/bench", [&](const httplib::Request&, httplib::Response& res) {
        if (!mu.try_lock_for(std::chrono::seconds(3))) {
            res.status = 409;
            res.set_content("{\"error\":\"busy: wait for the current reply or load to finish\"}", "application/json");
            return;
        }
        std::lock_guard<std::timed_mutex> lk(mu, std::adopt_lock);
        if (!eng.model) { res.status = 409; res.set_content("{\"error\":\"no model loaded\"}", "application/json"); return; }
        try {
            std::fprintf(stderr, "benchmark: reading 512 tokens, then generating 64\n");
            const double p = measure_prompt(512);
            const int draft = eng.spec ? eng.draft_max : 0;
            measure(4, draft);                                  // warm-up
            const double g = measure(64, draft);
            eng.model->reset();
            eng.cached.clear();
            std::fprintf(stderr, "benchmark: prompt %.1f tok/s, generation %.1f tok/s\n", p, g);
            res.set_content(dump({{"prompt_tps", std::round(p * 10) / 10}, {"gen_tps", std::round(g * 10) / 10},
                                  {"speculative", draft > 0}, {"draft_max", draft}, {"threads", eng.threads}}), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(dump({{"error", e.what()}}), "application/json");
        }
    });

    // ---------------- Speculative decoding on/off without reloading (when the draft layer is loaded) ----------------
    svr.Post("/api/spec", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) { res.status = 400; res.set_content("{\"error\":\"bad request\"}", "application/json"); return; }
        if (!mu.try_lock_for(std::chrono::seconds(3))) { res.status = 409; res.set_content("{\"error\":\"busy\"}", "application/json"); return; }
        std::lock_guard<std::timed_mutex> lk(mu, std::adopt_lock);
        if (!eng.model) { res.status = 409; res.set_content("{\"error\":\"no model loaded\"}", "application/json"); return; }
        const bool on = body.value("speculative", eng.spec != nullptr);
        const int d = std::min(8, std::max(1, body.value("draft_max", eng.draft_max > 0 ? eng.draft_max : 2)));
        if (on && !eng.model->has_mtp()) {
            res.status = 409;
            res.set_content(dump({{"error", "this model was loaded without its draft layer: turn on Speculative decoding on the setup screen and load it again"}}), "application/json");
            return;
        }
        eng.draft_max = d;
        eng.spec.reset();
        if (on) eng.spec = std::make_unique<Speculator>(*eng.model, d);
        eng.cached.clear();                                     // the draft block's cache starts over with the next reply
        eng.model->reset();
        std::fprintf(stderr, "speculative decoding %s%s\n", on ? "on, " : "off", on ? (std::to_string(d) + (d == 1 ? " guess per check" : " guesses per check")).c_str() : "");
        res.set_content(dump({{"ok", true}, {"speculative", on}, {"draft_max", d}}), "application/json");
    });

    // ---------------- Forget what was learned about the loaded model's experts ----------------
    svr.Post("/api/usage/reset", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::timed_mutex> lk(mu);
        std::fill(eng.usage.begin(), eng.usage.end(), 0.0f);
        std::error_code ec;
        if (!eng.usage_path.empty()) fs::remove(fs::u8path(eng.usage_path), ec);
        eng.usage_dirty = false;
        std::fprintf(stderr, "expert usage for this model forgotten; the next load places whole expert layers again\n");
        res.set_content("{\"ok\":true}", "application/json");
    });

    // ---------------- Loading progress and the console (no lock: they answer while a model loads) ----------------
    svr.Get("/api/progress", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(dump({{"loading", loading.load()}, {"progress", load_progress.load()}}), "application/json");
    });
    svr.Get("/api/log", [&](const httplib::Request& req, httplib::Response& res) {
        long long after = -1;
        if (req.has_param("after")) after = std::atoll(req.get_param_value("after").c_str());
        json lines = json::array();
        long long next = 0;
        {
            std::lock_guard<std::mutex> lk(g_log.mu);
            next = g_log.next;
            for (const auto& [n, t] : g_log.lines)
                if (n > after) lines.push_back({{"n", n}, {"text", t}, {"v", verbose_line(t)}});
        }
        res.set_content(dump({{"lines", lines}, {"next", next}}), "application/json");
    });

    // ---------------- OpenAI-compatible API ----------------
    const long long started = (long long)std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    // The loaded model's id: its path inside the models folder, as /v1/models lists it.
    auto loaded_id = [&]() -> std::string {
        if (eng.path.empty()) return {};
        std::error_code ec;
        auto rel = fs::u8path(eng.path).lexically_relative(fs::weakly_canonical(fs::u8path(a.models), ec));
        std::string r = rel.generic_u8string();
        return r.empty() || r.compare(0, 2, "..") == 0 ? fs::u8path(eng.path).filename().u8string() : r;
    };

    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json data = json::array();
        std::string cur;
        {
            std::unique_lock<std::timed_mutex> lk(mu, std::try_to_lock);
            cur = lk.owns_lock() ? loaded_id() : last_info.value("path", std::string());
            if (!lk.owns_lock() && !cur.empty()) cur = fs::u8path(cur).filename().u8string();
        }
        if (!cur.empty()) data.push_back({{"id", cur}, {"object", "model"}, {"created", started}, {"owned_by", "tierlm"}, {"loaded", true}});
        bool exists = false;
        for (const auto& m : scan_models(exists)) {
            if (!m.value("supported", false) || m.value("id", std::string()) == cur) continue;
            data.push_back({{"id", m["id"]}, {"object", "model"}, {"created", started}, {"owned_by", "tierlm"}, {"loaded", false}});
        }
        res.set_content(dump({{"object", "list"}, {"data", data}}), "application/json");
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("messages") || !body["messages"].is_array())
            return oai_error(res, 400, "expected a JSON body with a \"messages\" array");
        if (body.value("n", 1) != 1) return oai_error(res, 400, "only n = 1 is supported");
        // Wait for any reply (from the web page or another app) to finish: one at a time.
        auto lock = std::make_shared<std::unique_lock<std::timed_mutex>>(mu);
        // The model asked for: the loaded one if the name matches it, or is unknown (many apps send a fixed
        // name such as "gpt-4o"); a different model in the models folder is loaded first, with its saved settings.
        const std::string want = body.value("model", std::string());
        if (!want.empty() && want != loaded_id()) {
            std::error_code ec;
            const auto base = fs::weakly_canonical(fs::u8path(a.models), ec);
            const auto file = fs::weakly_canonical(fs::u8path(a.models) / fs::u8path(want), ec);
            const auto rel = file.lexically_relative(base);
            const bool in_folder = !ec && !rel.empty() && rel.begin()->string() != ".." && fs::is_regular_file(file, ec) &&
                                   file.extension() == ".gguf";
            if (in_folder) {
                try {
                    json saved = settings.value("models", json::object()).value(rel.generic_u8string(), json::object());
                    eng.draft_max = std::min(8, std::max(0, saved.value("draft_max", 0)));
                    load(file.u8string(), load_options(saved), saved.value("threads", a.threads), saved.value("seed", -1), saved.value("hot_experts", false));
                    settings["last_model"] = rel.generic_u8string();
                    write_settings(a.settings, settings);
                } catch (const std::exception& e) {
                    eng.spec.reset(); eng.model.reset(); eng.path.clear();
                    return oai_error(res, 500, std::string("could not load ") + want + ": " + e.what(), "server_error");
                }
            }
        }
        if (!eng.model)
            return oai_error(res, 400, "no model is loaded: load one in tierlm, or name one from /v1/models", "invalid_request_error", "model_not_found");

        Sampling sp;
        sp.temperature = body.value("temperature", sp.temperature);
        sp.top_p = body.value("top_p", sp.top_p);
        sp.top_k = body.value("top_k", sp.top_k);
        sp.min_p = body.value("min_p", sp.min_p);
        sp.repeat_penalty = body.value("repeat_penalty", 1.0f);   // off unless asked: it harms code, where repeats are normal
        sp.repeat_last_n = body.value("repeat_last_n", sp.repeat_last_n);
        int max_tokens = -1;
        if (body.contains("max_completion_tokens") && body["max_completion_tokens"].is_number()) max_tokens = body["max_completion_tokens"].get<int>();
        else if (body.contains("max_tokens") && body["max_tokens"].is_number()) max_tokens = body["max_tokens"].get<int>();
        std::vector<std::string> stops;
        if (body.contains("stop")) {
            if (body["stop"].is_string()) stops.push_back(body["stop"].get<std::string>());
            else if (body["stop"].is_array()) for (const auto& x : body["stop"]) if (x.is_string()) stops.push_back(x.get<std::string>());
        }
        if (body.contains("seed") && body["seed"].is_number_integer()) eng.rng.seed((uint32_t)body["seed"].get<long long>());
        const json messages = normalize_messages(body["messages"]);
        const bool stream = body.value("stream", false);
        const bool usage_in_stream = body.contains("stream_options") && body["stream_options"].is_object() &&
                                     body["stream_options"].value("include_usage", false);
        const std::string model_id = loaded_id();
        const long long created = (long long)std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        static std::atomic<unsigned> counter{0};
        const std::string id = "chatcmpl-tierlm-" + std::to_string(created) + "-" + std::to_string(++counter);
        auto finish = [](const std::string& stop) { return stop == "length" || stop == "context" ? "length" : "stop"; };

        if (!stream) {
            try {
                Gen g = generate(messages, sp, max_tokens, stops, nullptr, nullptr);
                res.set_content(dump({{"id", id}, {"object", "chat.completion"}, {"created", created}, {"model", model_id},
                    {"choices", json::array({{{"index", 0}, {"message", {{"role", "assistant"}, {"content", g.text}}},
                                              {"logprobs", nullptr}, {"finish_reason", finish(g.stop)}}})},
                    {"usage", {{"prompt_tokens", g.n_prompt}, {"completion_tokens", g.n_gen}, {"total_tokens", g.n_prompt + g.n_gen}}}}),
                    "application/json");
            } catch (const std::exception& e) {
                oai_error(res, 500, e.what(), "server_error");
            }
            return;
        }
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream", [&, messages, sp, max_tokens, stops, usage_in_stream, model_id, created, id, lock]
                                                              (size_t, httplib::DataSink& sink) {
            auto send = [&](const json& j) { std::string s = "data: " + dump(j) + "\n\n"; return sink.write(s.data(), s.size()); };
            auto chunk = [&](const json& delta, const json& fin) {
                return send({{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", model_id},
                             {"choices", json::array({{{"index", 0}, {"delta", delta}, {"logprobs", nullptr}, {"finish_reason", fin}}})}});
            };
            try {
                chunk({{"role", "assistant"}, {"content", ""}}, nullptr);
                Gen g = generate(messages, sp, max_tokens, stops, nullptr,
                                 [&](const std::string& delta, double, const std::vector<int32_t>&) {
                                     return delta.empty() ? true : chunk({{"content", delta}}, nullptr);
                                 });
                if (g.stop != "abort") {
                    chunk(json::object(), finish(g.stop));
                    if (usage_in_stream)
                        send({{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", model_id},
                              {"choices", json::array()},
                              {"usage", {{"prompt_tokens", g.n_prompt}, {"completion_tokens", g.n_gen}, {"total_tokens", g.n_prompt + g.n_gen}}}});
                }
            } catch (const std::exception& e) {
                send({{"error", {{"message", e.what()}, {"type", "server_error"}}}});
            }
            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            return true;
        });
    });

    // ---------------- network access and API key ----------------
    // By default tierlm only answers this computer. "Allow other devices" (saved in the settings) listens on the
    // network too, for phones, other PCs, or Open WebUI in Docker. An API key, if set, is then required by
    // everything that isn't plainly this computer, and by every /v1 request, so a tunnel (needed by Cursor)
    // can't open tierlm to the internet. --host and --api-key on the command line override the saved values.
    std::atomic<bool> allow_network{settings.value("allow_network", false)};
    std::string api_key = a.api_key_given ? a.api_key : settings.value("api_key", std::string());
    std::mutex net_mu;                   // guards api_key
    std::atomic<bool> relisten{false};
    auto bind_host = [&]() { return a.host_given ? a.host : (allow_network ? std::string("0.0.0.0") : std::string("127.0.0.1")); };

    svr.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
        const bool local = is_local(req);
        if (!local && !allow_network && !a.host_given) {
            oai_error(res, 403, "tierlm only accepts connections from this computer: turn on \"Allow other devices\" in tierlm's settings",
                      "permission_error");
            return httplib::Server::HandlerResponse::Handled;
        }
        std::string key;
        { std::lock_guard<std::mutex> lk(net_mu); key = api_key; }
        const bool is_api = req.path.compare(0, 4, "/v1/") == 0 || req.path.compare(0, 5, "/api/") == 0;
        const bool needs_key = !key.empty() && is_api && (!local || req.path.compare(0, 4, "/v1/") == 0);
        if (needs_key) {
            std::string auth = req.get_header_value("Authorization");
            std::string given = auth.compare(0, 7, "Bearer ") == 0 ? auth.substr(7) : req.get_header_value("X-Api-Key");
            if (!same_secret(given, key)) {
                res.set_header("WWW-Authenticate", "Bearer");
                oai_error(res, 401, "missing or wrong API key (send it as \"Authorization: Bearer <key>\")",
                          "authentication_error", "invalid_api_key");
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    auto network_info = [&](bool show_key) {
        std::string key;
        { std::lock_guard<std::mutex> lk(net_mu); key = api_key; }
        const std::string lan = lan_address();
        json urls = json::array();
        urls.push_back({{"label", "This computer"}, {"url", "http://127.0.0.1:" + std::to_string(a.port) + "/v1"}});
        if (!lan.empty()) urls.push_back({{"label", "Other devices on your network"}, {"url", "http://" + lan + ":" + std::to_string(a.port) + "/v1"}});
        json j = {{"allow_network", (bool)allow_network}, {"listening_on", bind_host()}, {"port", a.port}, {"lan_address", lan},
                  {"api_key_set", !key.empty()}, {"urls", urls}, {"fixed_by_command_line", a.host_given || a.api_key_given}};
        if (show_key) j["api_key"] = key;
        return j;
    };
    svr.Get("/api/network", [&](const httplib::Request& req, httplib::Response& res) {
        res.set_content(dump(network_info(is_local(req))), "application/json");   // the key itself only to this computer
    });
    // Change network access or the API key. Only from this computer: someone on the network mustn't be able to
    // open it wider or change the key.
    svr.Post("/api/network", [&](const httplib::Request& req, httplib::Response& res) {
        if (!is_local(req)) { oai_error(res, 403, "network settings can only be changed on the computer running tierlm", "permission_error"); return; }
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) { oai_error(res, 400, "bad request"); return; }
        const std::string before = bind_host();
        if (body.contains("allow_network")) { allow_network = body.value("allow_network", false); settings["allow_network"] = (bool)allow_network; }
        if (body.contains("api_key") && !a.api_key_given) {
            std::string k = body["api_key"].is_string() ? body["api_key"].get<std::string>() : std::string();
            if (k == "generate") k = random_key();
            { std::lock_guard<std::mutex> lk(net_mu); api_key = k; }
            settings["api_key"] = k;
        }
        write_settings(a.settings, settings);
        // Listen on the new address: stop the listener a moment after this reply is sent; main() starts it again.
        if (bind_host() != before) {
            relisten = true;
            std::thread([&] { std::this_thread::sleep_for(std::chrono::milliseconds(300)); svr.stop(); }).detach();
        }
        res.set_content(dump(network_info(true)), "application/json");
    });

    std::fprintf(stderr, "\n  tierlm is ready: open http://127.0.0.1:%d in your browser\n  %s\n  OpenAI-compatible API: http://127.0.0.1:%d/v1\n  (Ctrl+C to stop)\n\n",
                 a.port, eng.model ? "" : "No model loaded yet: the page will show the setup screen.", a.port);
    for (;;) {
        const std::string host = bind_host();
        if (host != "127.0.0.1" && host != "localhost") {
            const std::string lan = lan_address();
            std::fprintf(stderr, "  Other devices can connect: http://%s:%d (API: http://%s:%d/v1)%s\n\n", lan.empty() ? host.c_str() : lan.c_str(), a.port,
                         lan.empty() ? host.c_str() : lan.c_str(), a.port, api_key.empty() ? "  -- no API key set" : "");
        }
        relisten = false;
        if (!svr.listen(host, a.port)) {
            std::fprintf(stderr, "error: could not listen on %s:%d (is something else using that port? try --port 8041)\n", host.c_str(), a.port);
            return 1;
        }
        if (!relisten) break;
        std::fprintf(stderr, "  network setting changed: listening again\n");
    }
    return 0;
}
