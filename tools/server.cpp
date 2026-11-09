// tierlm-server: runs the engine behind a local web page with a live dashboard.
//
//   tierlm-server [model.gguf] [--port 8080] [--host 127.0.0.1] [-d auto|cpu|gpu|<device>] [-t threads] [-c context]
//                 [--web dir] [--models dir]
//   Without a model, the page opens a setup screen to choose the hardware and model.
//
// Endpoints
//   GET  /             the web page (web/index.html, re-read on every load so it can be edited live)
//   GET  /vendor/katex.min.js  the maths renderer the page uses (KaTeX, MIT licence, web/vendor)
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
//                     "min_p":0.05, "repeat_penalty":1.1, "repeat_last_n":64, "presence_penalty":0,
//                     "frequency_penalty":0, "max_tokens":-1}
//                     streams Server-Sent Events: one "prompt" event, a "token" event per token
//                     (with the experts picked in every layer), then "done"
//   POST /api/reset   clears the KV cache
//   GET  /api/network  API and network settings, and the addresses other apps should use
//   POST /api/network  {"allow_network": true, "api_key": "..."}  (this computer only; saved)
//
// OpenAI-compatible API, for apps such as Open WebUI, Continue or anything built on the OpenAI SDK
// (base URL http://<this computer>:<port>/v1; see docs/API.md):
//   GET  /v1/models            the loaded model and every supported model in the models folder
//   POST /v1/chat/completions  chat, streamed or not, with tool calling; naming another model loads it first
//   POST /v1/embeddings        embeddings from the embedding model loaded next to the chat model (/api/embed)
#include "model.h"
#include "speculate.h"
#include "gguf_split.h"
#include "tokenizer.h"
#include "encoder.h"
#include "vision.h"
#include "gguf.h"
#include "ggml-backend.h"
#include <list>
#include <map>
#include "httplib.h"
#include "json.hpp"
#include "agent.h"
#include "harmony.h"
#include "workspace.h"
#include "chats.h"
#include "download.h"

#include <climits>
#include <algorithm>
#include <functional>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "backends.h"
#include <deque>
#include <condition_variable>
#include <shared_mutex>
#ifdef _WIN32
#include <windows.h>
#include <shobjidl.h>
#include <shellapi.h>
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
enum class ChatFmt { Tulu, ChatML, Llama3, Mistral, Harmony, DeepSeek, Kimi, Plain };

const char* chat_fmt_name(ChatFmt f) {
    switch (f) {
        case ChatFmt::Tulu: return "tulu"; case ChatFmt::ChatML: return "chatml";
        case ChatFmt::Llama3: return "llama3"; case ChatFmt::Mistral: return "mistral"; case ChatFmt::Harmony: return "harmony";
        case ChatFmt::DeepSeek: return "deepseek"; case ChatFmt::Kimi: return "kimi";
        default: return "plain";
    }
}

ChatFmt detect_chat(const Tokenizer& tok) {
    const std::string& t = tok.chat_template();
    auto has = [&](const char* s) { return t.find(s) != std::string::npos; };
    if (has("<|channel|>")) return ChatFmt::Harmony;                 // gpt-oss (OpenAI's harmony format)
    if (has("<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>") || tok.find("<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>") >= 0) return ChatFmt::DeepSeek;   // <｜Assistant｜>
    if (has("<|im_middle|>") || tok.find("<|im_middle|>") >= 0) return ChatFmt::Kimi;
    if (has("<|im_start|>")) return ChatFmt::ChatML;
    if (has("<|start_header_id|>")) return ChatFmt::Llama3;
    if (has("<|user|>")) return ChatFmt::Tulu;
    if (has("[INST]")) return ChatFmt::Mistral;
    if (tok.find("<|channel|>") >= 0 && tok.find("<|return|>") >= 0) return ChatFmt::Harmony;
    if (tok.find("<|im_start|>") >= 0) return ChatFmt::ChatML;
    if (tok.find("<|start_header_id|>") >= 0) return ChatFmt::Llama3;
    if (tok.find("<|user|>") >= 0) return ChatFmt::Tulu;
    if (tok.find("[INST]") >= 0 || tok.spm()) return ChatFmt::Mistral;
    return ChatFmt::Plain;
}

// Tool calling in a conversation, made plain for format_chat: the tools go into the system message, an assistant
// message's tool calls are written out the way the model writes them (unless its text already has them, as the
// page sends it), and tool results become a user turn of <tool_response> blocks, as Qwen's chat templates do.
json with_tools(const json& messages, const agent::ojson& tools, agent::ToolFmt tf) {
    json out = json::array();
    const bool use = tools.is_array() && !tools.empty();
    size_t i = 0;
    std::string system;
    if (!messages.empty() && messages[0].value("role", std::string()) == "system") { system = messages[0].value("content", std::string()); i = 1; }
    if (use) out.push_back({{"role", "system"}, {"content", agent::system_with_tools(tf, system, tools)}});
    else if (i == 1) out.push_back(messages[0]);
    for (; i < messages.size(); ++i) {
        const json& m = messages[i];
        const std::string role = m.value("role", std::string("user"));
        if (role == "tool") {
            std::string c;
            for (; i < messages.size() && messages[i].value("role", std::string()) == "tool"; ++i)
                c += std::string(c.empty() ? "" : "\n") + "<tool_response>\n" + agent::trim(messages[i].value("content", std::string())) + "\n</tool_response>";
            --i;
            out.push_back({{"role", "user"}, {"content", c}});
        } else if (role == "assistant" && m.contains("tool_calls") && m["tool_calls"].is_array() && !m["tool_calls"].empty()) {
            std::string c = m.value("content", std::string());
            if (c.find(agent::kToolOpen) == std::string::npos) c += agent::render_tool_calls(tf, m["tool_calls"], !agent::trim(c).empty());
            out.push_back({{"role", "assistant"}, {"content", c}});
        } else out.push_back({{"role", role}, {"content", m.value("content", std::string())}});
    }
    return out;
}

// think: what the assistant turn starts with in models that can think (Qwen3 and later, as their chat templates
// do): 1 "<think>\n" (the model reasons first; only for templates that start it), 0 "<think>\n\n</think>\n\n" (thinking
// off: the answer straight away), -1 nothing (the model decides, or can't think).
std::string format_chat(const json& messages, ChatFmt f, const Tokenizer& tok, int think = -1) {
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
        for (const auto& m : messages) {
            const std::string role = m.value("role", std::string("user")), c = m.value("content", std::string());
            // With thinking off, earlier replies were written after an empty thinking block: show them the same way,
            // so the conversation reads exactly as the model saw it and the cache carries on through the last reply.
            const bool empty_think = think == 0 && role == "assistant" && c.compare(0, 7, "<think>") != 0;
            s += "<|im_start|>" + role + "\n" + (empty_think ? "<think>\n\n</think>\n\n" : "") + c + "<|im_end|>\n";
        }
        return s + "<|im_start|>assistant\n" + (think == 0 ? "<think>\n\n</think>\n\n" : think == 1 ? "<think>\n" : "");
    case ChatFmt::DeepSeek: {                             // DeepSeek V3 / R1 / V3.1: <｜User｜>... <｜Assistant｜>...
        const std::string U = "<\xEF\xBD\x9CUser\xEF\xBD\x9C>", A = "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>";
        const std::string bos = "<\xEF\xBD\x9C" "begin\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>";
        const std::string eos = "<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>";
        const std::string& ct = tok.chat_template();
        if (tok.find(bos) >= 0) s = bos;
        std::string sys;                                  // system prompts come first, joined
        for (const auto& m : messages) if (m.value("role", "") == "system") sys += (sys.empty() ? "" : "\n\n") + m.value("content", std::string());
        s += sys;
        for (const auto& m : messages) {
            const std::string role = m.value("role", "");
            std::string c = m.value("content", std::string());
            if (role == "user") s += U + c;
            else if (role == "assistant") {               // earlier answers without their reasoning, as the template does
                const size_t e = c.rfind("</think>");
                if (e != std::string::npos) c = c.substr(e + 8);
                s += A + c + eos;
            }
        }
        s += A;
        if (ct.find("<think>\\n") != std::string::npos && ct.find("thinking") == std::string::npos) return s + "<think>\n";   // R1: always thinks
        if (ct.find("</think>") != std::string::npos && ct.find("thinking") != std::string::npos)                         // V3.1: a switch
            return s + (think == 1 ? "<think>" : "</think>");
        return s;
    }
    case ChatFmt::Kimi: {                                 // Kimi K2: <|im_user|>user<|im_middle|>...<|im_end|>
        const std::string& ct = tok.chat_template();
        const char* kimi_sys = "You are Kimi, an AI assistant created by Moonshot AI.";
        bool have_sys = false;
        for (const auto& m : messages) have_sys |= m.value("role", "") == "system";
        if (!have_sys && ct.find(kimi_sys) != std::string::npos) s += std::string("<|im_system|>system<|im_middle|>") + kimi_sys + "<|im_end|>";
        for (const auto& m : messages) {
            const std::string role = m.value("role", std::string("user")), c = m.value("content", std::string());
            if (role == "system") s += "<|im_system|>system<|im_middle|>" + c + "<|im_end|>";
            else if (role == "assistant") s += "<|im_assistant|>assistant<|im_middle|>" + c + "<|im_end|>";
            else s += "<|im_user|>user<|im_middle|>" + c + "<|im_end|>";
        }
        return s + "<|im_assistant|>assistant<|im_middle|>";
    }
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
    float presence_penalty = 0.0f, frequency_penalty = 0.0f;   // OpenAI's: subtracted from a used token's logit
    int top_k = 40, repeat_last_n = 64;
};

// Penalties for tokens already used, then temperature with top-k, top-p (nucleus) and min-p filtering. All three
// penalties look at the last repeat_last_n tokens, as llama.cpp and LM Studio do: the repetition penalty divides a
// used token's logit (multiplies a negative one), then the presence penalty is taken off once and the frequency
// penalty once per use (Qwen recommends presence 1.5 for its instruct mode, to stop it repeating itself).
// temperature <= 0 means always take the most likely token (after the penalty).
// Only the top-k candidates matter, so one pass over the vocabulary keeps the best ones in a small heap, without
// copying the logits or computing every probability (gpt-oss has 201,088 tokens: that pass is most of the cost).
// The tokens sampling could pick, best first, with their weights (not normalised; greedy: the one best token), and
// how many of them top-p keeps (n). sample() draws from them; distribution() gives them as probabilities.
static std::vector<std::pair<float, int32_t>> candidates(const float* lg, size_t nv, const Sampling& s, const std::vector<int32_t>& recent,
                                                         size_t& n, double& cum) {
    // Tokens used in the last repeat_last_n positions, with their penalised logits (each token once).
    std::vector<std::pair<int32_t, float>> pen;
    const bool rep = s.repeat_penalty > 0.0f && s.repeat_penalty != 1.0f;
    if (rep || s.presence_penalty != 0.0f || s.frequency_penalty != 0.0f) {
        const size_t n = std::min<size_t>(recent.size(), (size_t)std::max(0, s.repeat_last_n));
        std::vector<std::pair<int32_t, int>> used;                 // each token once, with how often it was used
        for (size_t i = recent.size() - n; i < recent.size(); ++i) {
            const int32_t t = recent[i];
            if (t < 0 || (size_t)t >= nv) continue;
            auto it = std::find_if(used.begin(), used.end(), [t](auto& q) { return q.first == t; });
            if (it != used.end()) ++it->second; else used.push_back({t, 1});
        }
        for (const auto& [t, cnt] : used) {
            float v = lg[t];
            if (rep) v = v > 0 ? v / s.repeat_penalty : v * s.repeat_penalty;
            v -= s.presence_penalty + s.frequency_penalty * (float)cnt;
            pen.push_back({t, v});
        }
    }
    const bool greedy = s.temperature <= 0.0f;
    const size_t keep = greedy ? 1 : std::min<size_t>(s.top_k > 0 ? (size_t)s.top_k : 1024, nv);
    // The best keep + |pen| tokens by their logits before the penalty: after it, the best `keep` are among these plus
    // the penalised ones (a token outside them is beaten by at least `keep` tokens the penalty didn't touch).
    using C = std::pair<float, int32_t>;
    auto better = [](const C& a, const C& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); };
    const size_t want = std::min(nv, keep + pen.size());
    std::vector<C> h;
    h.reserve(want + pen.size());
    for (size_t i = 0; i < nv; ++i) {                 // h is a heap with its worst candidate first
        const float v = lg[i];
        if (h.size() < want) { h.push_back({v, (int32_t)i}); std::push_heap(h.begin(), h.end(), better); }
        else if (v > h.front().first) { std::pop_heap(h.begin(), h.end(), better); h.back() = {v, (int32_t)i}; std::push_heap(h.begin(), h.end(), better); }
    }
    for (const auto& [t, v] : pen) {
        auto it = std::find_if(h.begin(), h.end(), [t = t](const C& c) { return c.second == t; });
        if (it != h.end()) it->first = v; else h.push_back({v, t});
    }
    std::sort(h.begin(), h.end(), better);            // best first; equal logits: the lower token id, like max_element
    if (greedy) { n = 1; cum = 1.0; return {{1.0f, h[0].second}}; }
    if (h.size() > keep) h.resize(keep);
    const float mx = h[0].first;
    std::vector<std::pair<float, int32_t>> p(h.size());
    for (size_t i = 0; i < h.size(); ++i) p[i] = {std::exp((h[i].first - mx) / s.temperature), h[i].second};
    // min-p: drop tokens less than min_p times as likely as the best one (p[0].first is 1 after scaling).
    if (s.min_p > 0.0f) while (p.size() > 1 && p.back().first < s.min_p * p[0].first) p.pop_back();
    double total = 0;
    for (auto& x : p) total += x.first;
    cum = 0;
    n = 0;
    while (n < p.size() && cum < s.top_p * total) cum += p[n++].first;
    if (n == 0) n = 1, cum = p[0].first;
    return p;
}

int32_t sample(const float* lg, size_t nv, const Sampling& s, const std::vector<int32_t>& recent, std::mt19937& rng) {
    size_t n = 0;
    double cum = 0;
    const std::vector<std::pair<float, int32_t>> p = candidates(lg, nv, s, recent, n, cum);
    if (s.temperature <= 0.0f) return p[0].second;
    std::uniform_real_distribution<double> u(0.0, cum);
    double r = u(rng);
    for (size_t i = 0; i < n; ++i) { r -= p[i].first; if (r <= 0) return p[i].second; }
    return p[n - 1].second;
}

// The probabilities sample() picks with: (token, probability) for each token it could pick.
std::vector<std::pair<int32_t, float>> distribution(const float* lg, size_t nv, const Sampling& s, const std::vector<int32_t>& recent) {
    size_t n = 0;
    double cum = 0;
    const std::vector<std::pair<float, int32_t>> p = candidates(lg, nv, s, recent, n, cum);
    std::vector<std::pair<int32_t, float>> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = {p[i].second, (float)(p[i].first / cum)};
    return out;
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

// LM Studio keeps the embedding model it comes with (nomic-embed-text-v1.5, shown there as
// "text-embedding-nomic-embed-text-v1.5") apart from downloaded models, in .internal/bundled-models.
std::vector<std::string> bundled_model_folders() {
    std::vector<std::string> out;
#ifdef _WIN32
    const std::string home = env("USERPROFILE");
#else
    const std::string home = env("HOME");
#endif
    if (home.empty()) return out;
    for (const char* sub : { ".lmstudio/.internal/bundled-models", ".cache/lm-studio/.internal/bundled-models" }) {
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
    for (const auto& g : skipped_cuda_gpus())
        w.push_back({{"code", "cuda_skip"}, {"gpu", g.name}, {"cc", std::to_string(g.major) + "." + std::to_string(g.minor)},
                     {"message", g.name + " is a generation (compute capability " + std::to_string(g.major) + "." + std::to_string(g.minor) +
                                 ") that this tierlm's CUDA engine doesn't cover (it covers " + cuda_built_for() +
                                 " and newer), so it runs through Vulkan instead. It works, just more slowly than CUDA would."}});
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
// ---- several replies at once (Connections > 1) ----
// Each reply runs on its own thread, as with one, but asks the batcher for its steps: the batcher waits a moment for
// every running reply to ask, then computes all of them in one model step (Model::eval_seqs), so the weights are read
// once for all of them. A long prompt is fed in pieces of 256 tokens, so it doesn't hold up the others' words.
struct Batcher {
    std::mutex m;
    std::condition_variable cv, done_cv;
    struct Req { int seq; std::vector<int32_t> ids; std::vector<float> logits; std::vector<int32_t> routing; bool done = false; std::string err; };
    std::vector<Req*> pending;
    int active = 0;                      // replies running now (each asks for one step at a time)
    bool stop = false;
    Model* model = nullptr;
    const int* threads = nullptr;        // (the engine's setting, which Find the fastest settings changes)
    std::thread th;
    const bool log = !env("TIERLM_BATCH_LOG").empty();
    int st_steps = 0, st_reqs = 0;
    double st_ms = 0;
    std::chrono::steady_clock::time_point st_t0 = std::chrono::steady_clock::now();
    void start(Model* m_, const int* threads_) {
        model = m_; threads = threads_; stop = false; active = 0; pending.clear();
        th = std::thread([this] { loop(); });
    }
    void finish() {
        { std::lock_guard<std::mutex> lk(m); stop = true; }
        cv.notify_all();
        if (th.joinable()) th.join();
        model = nullptr;
    }
    ~Batcher() { finish(); }
    std::vector<float> eval(int seq, const std::vector<int32_t>& ids, std::vector<int32_t>* routing) {
        if (routing) routing->clear();
        std::vector<float> lg;
        for (size_t at = 0; at < ids.size();) {
            const size_t n = std::min<size_t>(256, ids.size() - at);
            Req r{seq, std::vector<int32_t>(ids.begin() + at, ids.begin() + at + n)};
            {
                std::unique_lock<std::mutex> lk(m);
                pending.push_back(&r);
                cv.notify_all();
                done_cv.wait(lk, [&] { return r.done; });
            }
            if (!r.err.empty()) throw std::runtime_error(r.err);
            if (routing) routing->insert(routing->end(), r.routing.begin(), r.routing.end());
            lg.swap(r.logits);
            at += n;
        }
        return lg;
    }
    void loop() {
        for (;;) {
            std::vector<Req*> batch;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return stop || !pending.empty(); });
                if (stop) {
                    for (Req* r : pending) { r->err = "the model was unloaded"; r->done = true; }
                    pending.clear();
                    done_cv.notify_all();
                    return;
                }
                // Wait for the other running replies to ask too: between steps they only pick their next word and
                // send it, well under a millisecond. Going without them made the replies take turns (one step each,
                // one after the other) instead of sharing steps, so 2 at once were no faster than 1. The time limit is
                // only for a reply held up by something else (a slow connection), so it can't stall the others long.
                const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(30);
                while ((int)pending.size() < active && !stop)
                    if (cv.wait_until(lk, until) == std::cv_status::timeout) break;
                batch.swap(pending);
            }
            // Always in conversation order: a step then looks the same as the last one when the same replies are
            // running, so the GPU can replay it (CUDA graphs need identical steps; with the order flipping between
            // steps they never kick in, and 2 replies ran at 123 tok/s in all where the engine does 262).
            std::sort(batch.begin(), batch.end(), [](const Req* x, const Req* y) { return x->seq < y->seq; });
            std::vector<Model::SeqInput> in;
            int n_tok = 0;
            for (Req* r : batch) { in.push_back({r->seq, r->ids}); n_tok += (int)r->ids.size(); }
            std::string err;
            std::vector<std::vector<float>> out;
            std::vector<int32_t> routing;
            const bool one_batch = n_tok <= model->batch();
            const auto c0 = std::chrono::steady_clock::now();
            try { out = model->eval_seqs(in, *threads, one_batch ? &routing : nullptr); }
            catch (const std::exception& e) { err = e.what(); }
            if (log) {                           // TIERLM_BATCH_LOG=1: how well the steps are shared, every 2 s
                const auto now = std::chrono::steady_clock::now();
                st_steps++; st_reqs += (int)batch.size(); st_ms += std::chrono::duration<double, std::milli>(now - c0).count();
                if (now - st_t0 > std::chrono::seconds(2)) {
                    const double wall = std::chrono::duration<double, std::milli>(now - st_t0).count();
                    std::fprintf(stderr, "batcher: %d steps, %.2f replies each, %.1f ms computing per step, %.0f%% of the time computing\n",
                                 st_steps, (double)st_reqs / st_steps, st_ms / st_steps, 100.0 * st_ms / wall);
                    st_steps = st_reqs = 0; st_ms = 0; st_t0 = now;
                }
            }
            {
                std::lock_guard<std::mutex> lk(m);
                const auto& hp = model->hparams();
                const size_t per = hp.is_moe() ? (size_t)hp.n_layer * hp.n_expert_used : 0;
                size_t t0 = 0;
                for (size_t i = 0; i < batch.size(); ++i) {
                    Req* r = batch[i];
                    if (!err.empty()) r->err = err;
                    else {
                        r->logits = std::move(out[i]);
                        if (one_batch && per) r->routing.assign(routing.begin() + t0 * per, routing.begin() + (t0 + r->ids.size()) * per);
                    }
                    t0 += r->ids.size();
                    r->done = true;
                }
            }
            done_cv.notify_all();
        }
    }
};

// FNV-1a: a quick 64-bit fingerprint (images: the same image sent again is recognised).
uint64_t fnv1a(const std::string& s, uint64_t h = 1469598103934665603ULL) {
    for (unsigned char ch : s) { h ^= ch; h *= 1099511628211ULL; }
    return h;
}

// What a reply holds while it runs: the engine lock (shared when several replies run at once, so they don't block
// each other but a model load waits for all of them) and, then, one of the connections. Released when it ends.
struct Hold {
    std::shared_timed_mutex* m = nullptr;
    bool shared = false;
    std::atomic<int>* free = nullptr;
    void release() {
        if (m) { if (shared) m->unlock_shared(); else m->unlock(); m = nullptr; }
        if (free) { ++*free; free = nullptr; }
    }
    ~Hold() { release(); }
};

struct Engine {
    // Several replies at once: one slot per connection (a conversation of the model, Model::eval_seqs), with the
    // tokens in its part of the cache. Empty: one reply at a time, as before.
    struct Slot { std::vector<int32_t> cached; bool busy = false; std::mt19937 rng{std::random_device{}()}; };
    std::vector<Slot> slots;
    std::mutex slots_mu;
    std::atomic<int> free_slots{0};
    int ctx_per = 0;                    // context of each connection (the cache holds ctx_per x connections)
    Batcher batch;
    std::mutex usage_mu;                // expert usage is learned from every reply, which may run at once
    std::unique_ptr<Speculator> spec;   // speculative decoding with the model's MTP draft block, when turned on
    int draft_max = 0;                  // most tokens the draft block guesses per check (0: 2 for MoE models, 3 for dense)
    float draft_p_min = 0.0f;           // stop guessing once the draft block is less sure than this (0: always guess draft_max)
    // Prompt lookup (guesses copied from earlier in the conversation): most tokens guessed per check; -1 chooses
    // for the model and how it's placed, 0 turns it off. `look` does it alone when the draft block is off.
    int lookup = -1;
    std::unique_ptr<Speculator> look;
    std::unique_ptr<Model> model;
    std::unique_ptr<Vision> vision;     // the model's eyes (Qwen3.5 / 3.6 with an mmproj file next to it)
    std::string vision_file;
    // Most image words per image (bigger images are scaled down; reading time grows faster than this). The page's
    // "Image detail" sends 512, 1024 or 2048 with each request; 0 (automatic): 2048 when a GPU reads images,
    // 1024 when only the CPU does.
    int image_tokens_for(int asked) const {
        if (asked > 0) return std::max(64, std::min(asked, 16384));
        return vision && (vision->on_gpu() || vision->gpu_helps()) ? 2048 : 1024;
    }
    // The image words of an image: (find or encode, then register with the model) its first id and size.
    struct Pic { int32_t first = 0; int n = 0; };
    Pic image(const std::string& bytes, int n_threads, int asked_tokens) {
        const int image_tokens = image_tokens_for(asked_tokens);
        if (!vision) throw std::runtime_error("this model can't read images: load Qwen3.5 / 3.6 with its mmproj (vision) file in the same folder");
        const uint64_t key = fnv1a(bytes, (uint64_t)image_tokens * 1000003ULL);
        int w = 0, h = 0;
        if (!image_size((const uint8_t*)bytes.data(), bytes.size(), &w, &h)) throw std::runtime_error("an image isn't in a format tierlm reads (PNG, JPEG, BMP, GIF)");
        const int n = vision->tokens_for(w, h, image_tokens);
        if (const int32_t f = model->find_image(key); f >= 0) return {f, n};
        const auto t0 = std::chrono::steady_clock::now();
        const Vision::Prepared p = vision->prepare(decode_image((const uint8_t*)bytes.data(), bytes.size()), image_tokens);
        std::vector<float> words = vision->encode(p, n_threads);
        const int32_t first = model->add_image(key, std::move(words), p.gh / vision->merge(), p.gw / vision->merge());
        std::fprintf(stderr, "image: %dx%d -> %dx%d, %d image words, read in %.2f s%s\n", w, h, p.width, p.height, p.tokens(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                     vision->gpu_helps() ? (vision->last_helped() ? " (CPU + GPU)" : " (CPU: no room on the GPU)") : "");
        return {first, p.tokens()};
    }
    Tokenizer tok;
    ChatFmt chat = ChatFmt::Plain;
    agent::ToolFmt tool_fmt = agent::ToolFmt::Hermes;   // how this model writes tool calls
    // Thinking (reasoning before the answer). think_switch: the model's chat template can turn it off (Qwen3 and
    // later: "enable_thinking"); think_starts: its template opens the <think> block itself, so the model's reply
    // begins inside it. thinking: on or off for this model (saved with its settings; a request can choose too).
    // gpt-oss always reasons; effort sets how much ("low", "medium", "high").
    bool think_switch = false, think_starts = false, thinking = true;
    std::string effort = "medium";
    std::string identity;                                // harmony models: the model identity line of their template
    std::vector<int32_t> cached;      // the tokens currently in the KV cache
    // Other conversations, saved to RAM when a request for a different one needed the cache, so they come back
    // without being read again. A coding agent's side requests (opencode titling a session, say) would otherwise
    // throw away its long instructions each time. Most recently used first.
    struct SavedConv {
        std::vector<int32_t> tokens;
        Model::Snapshot snap;
        bool has_spec = false;
        Speculator::State spec;
    };
    std::list<SavedConv> saved;
    size_t saved_bytes() const { size_t n = 0; for (const auto& c : saved) n += c.snap.bytes; return n; }
    // Room for saved conversations: up to 4 GB, and never more than a quarter of the RAM that is free now.
    size_t saved_budget() const {
        size_t budget = (size_t)4 << 30;
        if (ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(cpu, &free, &total);
            if (free > 0) budget = std::min(budget, free / 4 + saved_bytes());
        }
        return budget;
    }
    // Saves the conversation in the cache (when it's long enough to be worth it), replacing saved ones it extends.
    void save_current() {
        if (!model || cached.size() < 256 || model->n_past() != (int)cached.size()) return;
        for (auto it = saved.begin(); it != saved.end();) {
            const bool prefix = it->tokens.size() <= cached.size() && std::equal(it->tokens.begin(), it->tokens.end(), cached.begin());
            it = prefix ? saved.erase(it) : std::next(it);
        }
        SavedConv c;
        c.tokens = cached;
        c.snap = model->save();
        if (spec) { c.has_spec = true; c.spec = spec->state(); }
        saved.push_front(std::move(c));
        const size_t budget = saved_budget();
        while (saved.size() > 8 || (saved.size() > 1 && saved_bytes() > budget)) saved.pop_back();
        if (saved.size() == 1 && saved_bytes() > budget) saved.clear();
        std::fprintf(stderr, "cache: saved a %zu-token conversation to RAM (%.0f MB; %zu kept)\n", cached.size(),
                     saved.empty() ? 0.0 : saved.front().snap.bytes / 1048576.0, saved.size());
    }

    // How many tokens prompt lookup guesses at most. Checking guesses costs memory for the undo copies (hybrid
    // models keep one copy of their running memory per guess) and, with experts in RAM, reading more experts:
    // each guessed token picks its own. So shorter there, longer when everything is in VRAM.
    int lookup_len() const {
        if (!model || lookup == 0) return 0;
        if (lookup > 0) return std::min(lookup, 16);
        if (model->hparams().is_moe() && model->placement().split()) return 4;
        if (model->hparams().hybrid) return 6;
        return 8;
    }
    // The speculator replies use: the draft block's (which also does prompt lookup), or prompt lookup alone.
    Speculator* active_spec() { return spec ? spec.get() : look.get(); }
    // Call after `spec` or `lookup` changes.
    void spec_changed() {
        const int L = lookup_len();
        if (spec) spec->set_lookup(L);
        look.reset();
        if (!spec && L > 0 && model) { look = std::make_unique<Speculator>(*model, 0); look->set_lookup(L); }
    }
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
    std::vector<uint8_t> in_vram;        // [layer * E + expert]: in VRAM (hot experts or a whole layer), for the share below
    long long picks = 0, picks_vram = 0; // the router's picks since the last reply line, and how many VRAM served
    // Off while Find the fastest settings measures: its test questions must not teach the busiest-experts set, or
    // that set fits the test unusually well and wins against settings that serve real chats better.
    bool learn = true;
    void add_usage(const std::vector<int32_t>& routing, int K) {
        std::lock_guard<std::mutex> lk(usage_mu);
        if (usage.empty() || K <= 0) return;
        for (size_t i = 0; i < routing.size(); ++i) {
            const size_t l = (i / K) % hot_L;
            if (routing[i] >= 0 && routing[i] < hot_E) {
                if (learn) usage[l * hot_E + routing[i]] += 1.0f;
                if (!in_vram.empty()) { ++picks; picks_vram += in_vram[l * hot_E + routing[i]]; }
            }
        }
        if (learn) usage_dirty = true;
    }
    void save_usage(bool force = false) {
        std::lock_guard<std::mutex> lk(usage_mu);
        if (!usage_dirty || usage_path.empty() || (!force && since(usage_saved) < 10.0)) return;
        // Older habits fade: once the counts get large, halve them, so recent use weighs more.
        double tot = 0; for (float v : usage) tot += v;
        if (tot > 4e6) for (float& v : usage) v *= 0.5f;
        write_usage(usage_path, hot_L, hot_E, usage);
        usage_dirty = false;
        usage_saved = Clock::now();
    }
};

static std::string threads_text(int n) { return std::to_string(n) + (n == 1 ? " CPU thread" : " CPU threads"); }

// Header-only look at a GGUF file: what it is, how big, and whether tierlm can run it. header_only: the file is just
// the first megabytes of a model's first file (Get models), so its other parts aren't looked for.
json describe_model(const std::string& path, bool header_only = false) {
    json j = {{"file", std::filesystem::u8path(path).filename().u8string()}, {"supported", false}};
    std::error_code ec;
    j["bytes"] = (uint64_t)std::filesystem::file_size(std::filesystem::u8path(path), ec);
    // A model split over several files: count every part's size, and make sure they're all there.
    SplitInfo split;
    if (!header_only) try { split = gguf_split_info(path); } catch (const std::exception& e) { j["reason"] = e.what(); return j; }
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
    const bool mtp_arch = arch == "qwen35" || arch == "qwen35moe" || arch == "qwen3next" || arch == "qwen4exp";
    bool has_layer0 = false;                             // any layer-0 tensor (Flash-Next's layers have no attn_norm)
    for (int64_t k = 0; k < gguf_get_n_tensors(g) && !has_layer0; ++k) has_layer0 = std::strncmp(gguf_get_tensor_name(g, k), "blk.0.", 6) == 0;
    const bool draft_only = u32(arch + ".nextn_predict_layers") > 0 && !has_layer0;
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
    if (arch == "qwen35" || arch == "qwen35moe" || arch == "qwen3next" || arch == "qwen4exp") {
        const int64_t interval = u32(arch + ".full_attention_interval") ? u32(arch + ".full_attention_interval") : 4;
        kv_layers = layers / interval;
        j["hybrid"] = true;
    }
    if (layers && embd && heads) j["kv_bytes_per_token"] = 2 * kv_layers * (kv_heads ? kv_heads : heads) * head_dim * 2;
    if (arch == "deepseek2" && layers)                    // latent attention: one compressed vector per token per layer
        j["kv_bytes_per_token"] = kv_layers * (u32(arch + ".attention.kv_lora_rank") + u32(arch + ".rope.dimension_count")) * 2;
    const std::string tk = str("tokenizer.ggml.model");
    if (is_encoder_arch(arch)) {
        // An embedding model: it turns text into vectors for search and comparison, alongside a chat model.
        j["kind"] = "embedding";
        j["dims"] = embd;
        if (tk == "bert") { j["embedding"] = true; j["reason"] = "Embedding model: load it next to a chat model (side menu, Model info, Embedding model)"; }
        else j["reason"] = "embedding model with the " + (tk.empty() ? std::string("unknown") : tk) + " tokenizer, not supported yet";
        gguf_free(g);
        return j;
    }
    if (arch_supported(arch) && (tk == "gpt2" || tk == "llama")) j["supported"] = true;
    else if (arch_supported(arch)) j["reason"] = "the " + (tk.empty() ? std::string("unknown") : tk) + " tokenizer is not supported yet";
    else if (arch == "clip") {
        // A vision projector (mmproj file): the "eyes" of a vision model, not a model that runs on its own.
        j["kind"] = "mmproj";
        j["reason"] = "Vision file (mmproj): lets the model next to it read images. It loads with that model.";
    }
    else j["reason"] = "the " + (arch.empty() ? std::string("unknown") : arch) + " architecture is not supported yet";
    gguf_free(g);
    return j;
}

// The vision (mmproj) file for a model: a GGUF in the same folder with "mmproj" in its name that this engine reads
// and whose output width matches the model's (LM Studio and Hugging Face downloads keep them side by side, e.g.
// Qwen3.5-9B-Q4_K_M.gguf and mmproj-Qwen3.5-9B-F16.gguf). With several, the name closest to the model's wins, then
// F16 / BF16 over smaller ones (the vision part is small, and suffers most from rounding).
fs::path vision_file_for(const std::string& model_path, int n_embd) {
    std::error_code ec;
    const fs::path mp = fs::u8path(model_path), dir = mp.parent_path();
    auto lower = [](std::string x) { for (char& ch : x) ch = (char)std::tolower((unsigned char)ch); return x; };
    const std::string stem = lower(mp.stem().u8string());
    fs::path best;
    int best_score = -1;
    for (const auto& f : fs::directory_iterator(dir, ec)) {
        if (!f.is_regular_file(ec) || f.path().extension() != ".gguf") continue;
        const std::string name = lower(f.path().stem().u8string());
        if (name.find("mmproj") == std::string::npos) continue;
        if (!is_vision_file(f.path().u8string())) continue;
        gguf_init_params gp{ true, nullptr };
        if (gguf_context* g = gguf_init_from_file(f.path().u8string().c_str(), gp)) {
            const int64_t id = gguf_find_key(g, "clip.vision.projection_dim");
            const int64_t dim = id >= 0 ? (int64_t)gguf_get_val_u32(g, id) : 0;
            gguf_free(g);
            if (n_embd > 0 && dim > 0 && dim != n_embd) continue;          // another model's eyes
        }
        std::string bare = name;
        for (const char* cut : { "mmproj-", "mmproj_", "-mmproj", "_mmproj", "mmproj" })
            if (const size_t at = bare.find(cut); at != std::string::npos) { bare.erase(at, std::strlen(cut)); break; }
        size_t common = 0;
        while (common < bare.size() && common < stem.size() && bare[common] == stem[common]) ++common;
        const int prec = name.find("f16") != std::string::npos ? 3 : name.find("f32") != std::string::npos ? 2 : name.find("q8") != std::string::npos ? 1 : 0;
        const int score = (int)common * 4 + prec;
        if (score > best_score) { best_score = score; best = f.path(); }
    }
    return best;
}

// ---- OpenAI-compatible API helpers ----

// Images in a conversation. A message may carry "images" (data: URLs or plain base64 of PNG / JPEG / ...); in its
// text, each image's place is kPictureMark (the page puts them before the text; the API where they were sent).
// For Qwen3.5 / 3.6 the mark is what their chat template writes for an image; its <|image_pad|> token is replaced by
// the image's words just before the model reads the prompt.
const char* const kPictureMark = "<|vision_start|><|image_pad|><|vision_end|>";

// Base64 (standard or URL-safe; whitespace and padding ignored) to bytes. Empty if it isn't base64.
std::string base64_decode(const std::string& in) {
    auto val = [](unsigned char ch) -> int {
        if (ch >= 'A' && ch <= 'Z') return ch - 'A';
        if (ch >= 'a' && ch <= 'z') return ch - 'a' + 26;
        if (ch >= '0' && ch <= '9') return ch - '0' + 52;
        if (ch == '+' || ch == '-') return 62;
        if (ch == '/' || ch == '_') return 63;
        return -1;
    };
    std::string out;
    out.reserve(in.size() * 3 / 4);
    uint32_t acc = 0; int bits = 0;
    for (unsigned char ch : in) {
        if (ch == '=' || ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t') continue;
        const int v = val(ch);
        if (v < 0) return std::string();
        acc = acc << 6 | (uint32_t)v; bits += 6;
        if (bits >= 8) { bits -= 8; out += (char)(acc >> bits & 0xFF); }
    }
    return out;
}

// An image as sent ("data:image/png;base64,...." or plain base64) to its bytes. Throws with a message for the user.
std::string picture_bytes(const std::string& url) {
    std::string b64 = url;
    if (url.rfind("data:", 0) == 0) {
        const size_t comma = url.find(',');
        if (comma == std::string::npos || url.substr(0, comma).find(";base64") == std::string::npos)
            throw std::runtime_error("images must be sent as base64 data URLs (data:image/png;base64,...)");
        b64 = url.substr(comma + 1);
    } else if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
        throw std::runtime_error("tierlm doesn't fetch images from the web: send them as data URLs (data:image/png;base64,...)");
    }
    std::string bytes = base64_decode(b64);
    if (bytes.empty()) throw std::runtime_error("an image isn't valid base64");
    return bytes;
}

// The images of a conversation, in order, with each message's marks in place (a message whose text has fewer
// marks than images gets the missing ones first, as the page sends them). The "images" fields are removed.
std::vector<std::string> take_pictures(json& messages) {
    std::vector<std::string> pics;
    for (auto& m : messages) {
        if (!m.is_object() || !m.contains("images")) continue;
        const json imgs = m["images"];
        m.erase("images");
        if (!imgs.is_array() || imgs.empty()) continue;
        std::string text = m.contains("content") && m["content"].is_string() ? m["content"].get<std::string>() : std::string();
        size_t marks = 0;
        for (size_t at = text.find(kPictureMark); at != std::string::npos; at = text.find(kPictureMark, at + 1)) ++marks;
        std::string lead;
        for (size_t i = marks; i < imgs.size(); ++i) lead += kPictureMark;
        if (!lead.empty()) text = lead + (text.empty() ? "" : "\n") + text;
        m["content"] = text;
        for (const auto& u : imgs) if (u.is_string()) pics.push_back(picture_bytes(u.get<std::string>()));
    }
    return pics;
}


// OpenAI messages to plain {role, content} ones. Content may be a string, a list of parts
// ([{"type": "text", "text": ...}], images are skipped: no vision yet) or null. "developer" is OpenAI's newer
// name for "system". Tool calls ({"function": {"name", "arguments": "<json>"}}) are kept as {"name", "arguments": {...}},
// and tool results as role "tool".
json normalize_messages(const json& in) {
    json out = json::array();
    for (const auto& m : in) {
        if (!m.is_object()) continue;
        std::string role = m.value("role", std::string("user")), text;
        const json c = m.contains("content") ? m["content"] : json();
        json images = json::array();
        if (c.is_string()) text = c.get<std::string>();
        else if (c.is_array())
            for (const auto& part : c) {
                if (part.is_string()) { text += part.get<std::string>(); continue; }
                if (!part.is_object()) continue;
                const std::string type = part.value("type", std::string());
                if (type == "text" || type == "input_text") text += part.value("text", std::string());
                else if (type == "image_url" || type == "input_image" || type == "image") {
                    // {"image_url": {"url": "data:..."}} (OpenAI chat), {"image_url": "data:..."} (OpenAI responses),
                    // {"source": {"data": "...", "media_type": ...}} (Anthropic-style)
                    std::string url;
                    const json iu = part.contains("image_url") ? part["image_url"] : json();
                    if (iu.is_object()) url = iu.value("url", std::string());
                    else if (iu.is_string()) url = iu.get<std::string>();
                    else if (part.contains("source") && part["source"].is_object()) url = part["source"].value("data", std::string());
                    else if (part.contains("data") && part["data"].is_string()) url = part["data"].get<std::string>();
                    if (url.empty()) continue;
                    images.push_back(url);
                    text += kPictureMark;                          // where the image goes
                }
            }
        if (role == "developer") role = "system";
        if (role == "function") role = "tool";
        if (role != "system" && role != "user" && role != "assistant" && role != "tool") role = "user";
        // Thinking sent back separately (reasoning_content) goes back where the model wrote it.
        if (role == "assistant" && m.contains("reasoning_content") && m["reasoning_content"].is_string() &&
            !m["reasoning_content"].get<std::string>().empty() && text.find("<think>") == std::string::npos)
            text = "<think>\n" + m["reasoning_content"].get<std::string>() + "\n</think>\n\n" + text;
        json o = {{"role", role}, {"content", text}};
        if (!images.empty()) o["images"] = images;
        if (role == "assistant" && m.contains("tool_calls") && m["tool_calls"].is_array()) {
            json calls = json::array();
            for (const auto& tc : m["tool_calls"]) {
                const json f = agent::tool_function(tc);
                if (!f.is_object()) continue;
                json args = f.contains("arguments") ? f["arguments"] : json::object();
                json call = {{"name", f.value("name", std::string())}};
                if (args.is_string()) {                            // as sent: keeps the order the model wrote them in
                    call["arguments_json"] = args;
                    json p = json::parse(args.get<std::string>(), nullptr, false);
                    args = p.is_object() ? p : json::object();
                }
                call["arguments"] = args.is_object() ? args : json::object();
                calls.push_back(call);
            }
            if (!calls.empty()) o["tool_calls"] = calls;
        }
        out.push_back(o);
    }
    return out;
}

// Standard base64 (for embeddings sent as "encoding_format": "base64": little-endian float32s).
std::string base64_encode(const unsigned char* p, size_t n) {
    static const char* k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t)p[i] << 16 | (i + 1 < n ? (uint32_t)p[i + 1] << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        out += k[v >> 18 & 63]; out += k[v >> 12 & 63];
        out += i + 1 < n ? k[v >> 6 & 63] : '=';
        out += i + 2 < n ? k[v & 63] : '=';
    }
    return out;
}

void oai_error(httplib::Response& res, int status, const std::string& msg, const char* type = "invalid_request_error",
               const char* code = nullptr) {
    res.status = status;
    json e = {{"message", msg}, {"type", type}, {"param", nullptr}, {"code", code ? json(code) : json(nullptr)}};
    res.set_content(json({{"error", e}}).dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
}

// A request from this computer: a loopback address, and not passed on by a tunnel or proxy (Cloudflare Tunnel,
// ngrok and friends connect from localhost but add one of these headers), so it can't be someone outside.
// What another device may not do (see the pre-routing handler): anything but reading under /api/, except
// writing a reply (/api/chat) and starting a new chat (/api/reset); and a few reads that are about the computer
// rather than the model (the console, the hardware, the models folder's files).
bool remote_refused(const httplib::Request& req) {
    const std::string& p = req.path;
    if (p.compare(0, 5, "/api/") != 0 || p.compare(0, 8, "/api/v0/") == 0) return false;   // the page, /v1, LM Studio's API
    if (req.method == "GET" || req.method == "HEAD")
        return p == "/api/log" || p == "/api/devices" || p == "/api/models";
    return !(p == "/api/chat" || p == "/api/reset");
}

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
// Every line gets the local time it was printed (HH:MM:SS), in the black window and on the page.
std::string clock_now() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char b[16];
    std::strftime(b, sizeof b, "%H:%M:%S", &tm);
    return b;
}
struct LogRing {
    struct Line { long long n; std::string time, text; };
    std::mutex mu;
    std::deque<Line> lines;
    long long next = 0;
    void add(std::string time, std::string t) {
        while (!t.empty() && (t.back() == '\r' || t.back() == '\n')) t.pop_back();
        std::lock_guard<std::mutex> lk(mu);
        lines.push_back({next++, std::move(time), std::move(t)});
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
    // (TIERLM_NO_TIMES=1: the black window without the times, as before 0.8.5)
    const bool times = !std::getenv("TIERLM_NO_TIMES");
    std::thread([fd = fds[0], orig, rd, wr, times] {
        std::string part, part_time;
        bool line_start = true;                          // the next byte begins a line in the black window
        char buf[4096];
        for (;;) {
            const int n = rd(fd, buf, sizeof buf);
            if (n <= 0) break;
            const std::string now = clock_now();
            if (times) {
                // the time in front of each line, written as the line starts (text without a newline yet, such
                // as a progress line, still shows at once)
                std::string out;
                out.reserve((size_t)n + 32);
                for (int i = 0; i < n; ++i) {
                    if (line_start && buf[i] != '\r' && buf[i] != '\n') { out += now; out += "  "; line_start = false; }
                    out += buf[i];
                    if (buf[i] == '\n') line_start = true;
                }
                wr(orig, out.data(), (unsigned)out.size());
            } else {
                wr(orig, buf, (unsigned)n);
            }
            if (part.empty()) part_time = now;
            part.append(buf, (size_t)n);
            size_t p;
            while ((p = part.find('\n')) != std::string::npos) {
                g_log.add(part_time, part.substr(0, p));
                part.erase(0, p + 1);
                part_time = now;
            }
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
    std::shared_timed_mutex mu;                 // one load or generation at a time
    // Requests for a reply or a load wait their turn in line, first come first served (a mutex alone lets any
    // waiter in next, so a request could wait forever behind newer ones). Only replies and loads queue; Find the
    // fastest settings, the benchmark and setting changes still answer "busy" at once.
    struct Line {
        std::mutex m;
        std::condition_variable cv;
        std::deque<uint64_t> q;
        uint64_t next = 0;
        size_t waiting() { std::lock_guard<std::mutex> l(m); return q.size(); }
        // Waits until this request is first in line and the engine is free, then holds `mu` (the caller adopts
        // it). on_wait(ahead) is called whenever the number of requests ahead changes (0: only the one running).
        // Gives up (false) when gone() says the client left, or after max_wait.
        bool enter(std::shared_timed_mutex& mu, const std::function<bool()>& gone, const std::function<void(int)>& on_wait = nullptr,
                   std::chrono::seconds max_wait = std::chrono::seconds(1800)) {
            std::unique_lock<std::mutex> l(m);
            const uint64_t me = next++;
            q.push_back(me);
            const auto until = std::chrono::steady_clock::now() + max_wait;
            int shown = -1;
            for (;;) {
                if (q.front() == me && mu.try_lock()) { q.pop_front(); cv.notify_all(); return true; }
                const int ahead = (int)(std::find(q.begin(), q.end(), me) - q.begin());
                if (on_wait && ahead != shown) { shown = ahead; l.unlock(); on_wait(ahead); l.lock(); }
                if ((gone && gone()) || std::chrono::steady_clock::now() > until) {
                    q.erase(std::find(q.begin(), q.end(), me));
                    cv.notify_all();
                    return false;
                }
                cv.wait_for(l, std::chrono::milliseconds(50));   // also polls: mu is freed without telling the line
            }
        }
        // For a reply: like enter, but with several connections (slots() gives their free count, or null with one)
        // it takes the lock shared and one free connection, so up to that many replies run at once.
        bool enter_reply(std::shared_timed_mutex& mu, Hold& hold, const std::function<std::atomic<int>*()>& slots,
                         const std::function<bool()>& gone, const std::function<void(int)>& on_wait = nullptr,
                         std::chrono::seconds max_wait = std::chrono::seconds(1800)) {
            std::unique_lock<std::mutex> l(m);
            const uint64_t me = next++;
            q.push_back(me);
            const auto until = std::chrono::steady_clock::now() + max_wait;
            int shown = -1;
            for (;;) {
                if (q.front() == me) {
                    std::atomic<int>* free = slots ? slots() : nullptr;
                    if (!free) {
                        if (mu.try_lock()) { hold.m = &mu; hold.shared = false; q.pop_front(); cv.notify_all(); return true; }
                    } else if (mu.try_lock_shared()) {
                        int f = free->load();
                        while (f > 0 && !free->compare_exchange_weak(f, f - 1)) {}
                        if (f > 0) { hold.m = &mu; hold.shared = true; hold.free = free; q.pop_front(); cv.notify_all(); return true; }
                        mu.unlock_shared();
                    }
                }
                const int ahead = (int)(std::find(q.begin(), q.end(), me) - q.begin());
                if (on_wait && ahead != shown) { shown = ahead; l.unlock(); on_wait(ahead); l.lock(); }
                if ((gone && gone()) || std::chrono::steady_clock::now() > until) {
                    q.erase(std::find(q.begin(), q.end(), me));
                    cv.notify_all();
                    return false;
                }
                cv.wait_for(l, std::chrono::milliseconds(20));
            }
        }
    } line;
    auto reply_slots = [&]() -> std::atomic<int>* { return eng.slots.empty() ? nullptr : &eng.free_slots; };
    std::atomic<bool> loading{false};    // a model is being loaded (for the page's progress bar)
    std::atomic<float> load_progress{0.0f};

    // tierlm's own models folder: "models" next to it (or --models). Downloads always go there, and its models are
    // always listed (ids "@tierlm/<path>"), whichever folder is chosen for browsing (LM Studio's, say).
    std::error_code own_ec;
    const fs::path own_models = fs::absolute(fs::u8path(a.models_given ? a.models : std::string("models")), own_ec).lexically_normal();
    // The models folder: --models if given, else the one saved last time, else "models".
    json settings = read_settings(a.settings);
    bool recursive = settings.value("recursive", true);
    if (!a.models_given && settings.contains("models_folder") && settings["models_folder"].is_string()) {
        std::error_code ec;
        const std::string saved = settings["models_folder"];
        if (fs::is_directory(fs::u8path(saved), ec)) a.models = saved;
    }
    std::fprintf(stderr, "tierlm %s\nmodels folder: %s\n", TIERLM_VERSION, a.models.c_str());
    // A model file's id: its path inside the chosen folder, or "@tierlm/" + its path inside tierlm's own folder.
    auto model_key = [&](const std::string& p) -> std::string {
        std::error_code ec;
        const fs::path file = fs::weakly_canonical(fs::u8path(p), ec);
        auto inside = [&](const fs::path& root, std::string& out) {
            std::error_code e2;
            const std::string r = file.lexically_relative(fs::weakly_canonical(root, e2)).generic_u8string();
            if (e2 || r.empty() || r.compare(0, 2, "..") == 0) return false;
            out = r; return true;
        };
        std::string r;
        if (inside(fs::u8path(a.models), r)) return r;
        if (inside(own_models, r)) return "@tierlm/" + r;
        return fs::u8path(p).filename().u8string();
    };

    // hot: keep the model's busiest experts in VRAM, from what tierlm learned about it before (MoE models, experts in RAM).
    auto load = [&](const std::string& path, const LoadOptions& opt, int threads, int seed, bool hot = true) {
        json info = describe_model(path);
        if (!info.value("supported", false)) throw std::runtime_error(info.value("reason", std::string("unsupported model")));
        eng.save_usage(true);            // what the previous model learned
        eng.batch.finish();              // no replies run during a load (it holds the lock), so nothing is waiting
        eng.slots.clear(); eng.free_slots = 0;
        eng.spec.reset();
        eng.look.reset();
        eng.vision.reset(); eng.vision_file.clear();
        eng.model.reset();               // free the old model (and its GPU memory) first
        eng.cached.clear();
        eng.saved.clear();
        auto t0 = Clock::now();
        eng.tok = Tokenizer::from_gguf(path);
        LoadOptions o = opt;
        const int uL = info.value("layers", 0), uE = info.value("experts", 0);
        eng.usage.clear(); eng.usage_path.clear(); eng.usage_dirty = false; eng.hot_L = uL; eng.hot_E = uE;
        if (uE > 0 && uL > 0) {
            eng.usage_path = usage_file(a.settings, path);
            if (!read_usage(eng.usage_path, uL, uE, eng.usage)) eng.usage.assign((size_t)uL * uE, 0.0f);
            double tot = 0; for (float v : eng.usage) tot += v;
            if ((hot || o.live_hot) && tot > 0) o.expert_usage = eng.usage;
            else if (o.live_hot) {       // nothing learned yet: an even start (each layer the same share), the conversation does the rest
                o.expert_usage.assign((size_t)uL * uE, 0.0f);
                for (int l = 0; l < uL; ++l) for (int e = 0; e < uE; ++e) o.expert_usage[(size_t)l * uE + e] = 1.0f - 1e-4f * e;
            }
            eng.usage_saved = Clock::now();
        }
        // the VRAM this model needs beyond its weights, measured at the first load (see LoadOptions::vram_reserve_mb)
        if (o.vram_note.empty()) o.vram_note = usage_file(a.settings, path) + ".vram";
        o.progress = [&](float f) { load_progress = f; };
        load_progress = 0.0f;
        loading = true;
        struct Done { std::atomic<bool>& b; ~Done() { b = false; } } done{loading};
        eng.model = Model::load(path, o);
        eng.ctx_per = eng.model->max_ctx();
        if (o.parallel > 1) {            // several connections: each gets its share of the cache
            eng.slots.resize(o.parallel);
            eng.free_slots = o.parallel;
            eng.ctx_per = eng.model->max_ctx() / o.parallel;
            eng.batch.start(eng.model.get(), &eng.threads);
            std::fprintf(stderr, "connections: %d at once, %d tokens of context each\n", o.parallel, eng.ctx_per);
        }
        // Each guess an MoE model checks pulls in its own experts, so shorter guesses pay off there
        // (Qwen3.6-35B-A3B, RTX 3070 laptop: 35 tok/s with 1 or 2, 28 with 3); dense models do best with 3.
        if (eng.draft_max <= 0) eng.draft_max = eng.model->hparams().is_moe() ? 2 : 3;
        if (opt.mtp && eng.model->has_mtp()) eng.spec = std::make_unique<Speculator>(*eng.model, eng.draft_max, eng.draft_p_min);
        eng.spec_changed();
        // Its eyes: an mmproj (vision) file in the same folder, for models that read images (Qwen3.5 / 3.6).
        // (Not Flash-Next yet: its images aren't tested, so it runs text only and doesn't say it reads images.)
        if (eng.model->has_positions3() && !eng.model->hparams().any_ple()) {
            const fs::path vf = vision_file_for(path, eng.model->hparams().n_embd);
            if (!vf.empty()) {
                try {
                    // with the chat model's own GPU (on it, or helping), never another chip
                    const auto& gp = eng.model->placement().gpus;
                    const std::string where = opt.device == "cpu" || !eng.model->on_gpu() ? "cpu"
                                            : gp.empty() ? std::string("auto") : "auto:" + gp[0].name;
                    eng.vision = Vision::load(vf.u8string(), where);
                    eng.vision_file = vf.filename().u8string();
                    std::fprintf(stderr, "vision: %s on %s (%.0f MB)\n", eng.vision_file.c_str(), eng.vision->device_name().c_str(),
                                 eng.vision->weight_bytes() / 1048576.0);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "vision: %s could not be loaded: %s\n", vf.filename().u8string().c_str(), e.what());
                }
            }
        }
        eng.chat = detect_chat(eng.tok);
        eng.tool_fmt = agent::detect_tool_fmt(eng.tok.chat_template());
        eng.identity = harmony::identity_of(eng.tok.chat_template());
        {
            const std::string& ct = eng.tok.chat_template();
            eng.think_switch = eng.chat == ChatFmt::ChatML && ct.find("enable_thinking") != std::string::npos;
            eng.think_starts = eng.think_switch &&       // (its generation prompt: {{- '<think>\\n' }}, as Qwen3.5 and later)
                std::regex_search(ct, std::regex(R"('<think>(\\n|\n)'\s*\}\})"));
            if (eng.chat == ChatFmt::DeepSeek) {          // V3.1 has a thinking switch; R1 always starts by thinking
                eng.think_switch = ct.find("</think>") != std::string::npos && ct.find("thinking") != std::string::npos;
                eng.think_starts = eng.think_switch || ct.find("<think>\\n") != std::string::npos;
            }
            const json ms = settings.value("models", json::object());
            std::error_code ec;
            const std::string rel = model_key(path);
            const json saved = ms.contains(rel) && ms[rel].is_object() ? ms[rel] : json::object();
            eng.thinking = saved.value("thinking", false);   // off unless turned on (quicker replies; apps can still ask)
            eng.effort = saved.value("effort", std::string("medium"));
        }
        eng.path = path;
        eng.name = gguf_string(path, "general.name");
        eng.threads = threads > 0 ? threads : a.threads;
        eng.seed = seed;
        if (seed >= 0) eng.rng.seed((uint32_t)seed);
        eng.load_s = since(t0);
        const Placement& pl = eng.model->placement();
        std::string where;                      // how much is in RAM: it decides the speed, and it can change with the context
        if (pl.all_on_gpu()) where = ", all on the GPU";
        else if (eng.model->on_gpu() && pl.cpu_weights > 0) {
            char b[160];
            std::snprintf(b, sizeof b, ", %.1f GB in RAM (%d of %d layers' attention and %d of %d %s in VRAM)", pl.cpu_weights / 1e9,
                          pl.attn_gpu, pl.layers, pl.ffn_gpu, pl.layers, eng.model->hparams().is_moe() ? "layers' experts" : "feed-forwards");
            where = b;
        }
        std::fprintf(stderr, "loaded %s (%s, %s chat format) on %s%s in %.1fs\n", path.c_str(), eng.model->arch().c_str(),
                     chat_fmt_name(eng.chat), eng.model->device_name().c_str(), where.c_str(), eng.load_s);
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
            eng.in_vram.assign((size_t)uL * uE, 0);
            for (int l = 0; l < uL && (size_t)l < pl.hot.size(); ++l) {
                if (l < (int)pl.ffn_vram.size() && pl.ffn_vram[l]) for (int e = 0; e < uE; ++e) eng.in_vram[(size_t)l * uE + e] = 1;
                else for (int e : pl.hot[l]) if (e >= 0 && e < uE) eng.in_vram[(size_t)l * uE + e] = 1;
            }
            eng.picks = eng.picks_vram = 0;
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
        o.live_hot = body.value("live_experts", false);  // VRAM experts follow the conversation (see LoadOptions::live_hot)
        // The SSD tier: the folder chosen in the setup screen (for every model), and whether to use it ("ssd": false
        // keeps everything in RAM, as before 0.8.2).
        o.ssd_dir = settings.value("ssd_dir", std::string());
        if (body.contains("ssd")) o.ssd = body.value("ssd", true) ? -1 : 0;
        // Connections: replies written at once. The context length is per connection, so the cache holds that many.
        o.parallel = std::max(1, std::min(64, body.value("connections", 1)));
        if (o.parallel > 1 && o.max_ctx > 0) o.max_ctx *= o.parallel;
        return o;
    };
    if (!a.model.empty()) {
        LoadOptions o; o.max_ctx = a.ctx; o.device = a.device;
        try { load(a.model, o, a.threads, -1); }
        catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
    }

    httplib::Server svr;
    // Requests waiting in line each hold a thread, so the pool is bigger than httplib's default (the CPU count):
    // with 8 apps waiting, the page and status requests must still get through.
    svr.new_task_queue = [] { return new httplib::ThreadPool(32, 256); };

    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        std::string page = read_file(a.web + "/index.html");
        if (page.empty()) {
            res.status = 404;
            res.set_content("web/index.html not found. Run the server from the tierlm folder, or pass --web <folder>.", "text/plain");
            return;
        }
        res.set_content(page, "text/html; charset=utf-8");
    });
    // The one script the page loads besides itself (bundled, so maths shows without the internet). A fixed name:
    // nothing else in the folder is served.
    svr.Get("/vendor/katex.min.js", [&](const httplib::Request&, httplib::Response& res) {
        std::string js = read_file(a.web + "/vendor/katex.min.js");
        if (js.empty()) { res.status = 404; return; }
        res.set_header("Cache-Control", "max-age=86400");
        res.set_content(js, "text/javascript; charset=utf-8");
    });

    json last_info = {{"loaded", false}, {"version", TIERLM_VERSION}};
    svr.Get("/api/info", [&](const httplib::Request&, httplib::Response& res) {
        // While a reply is streaming or a model is loading, answer from the last snapshot instead of waiting.
        std::unique_lock<std::shared_timed_mutex> lk(mu, std::try_to_lock);
        if (!lk.owns_lock()) {
            json j = last_info; j["busy"] = true; j["waiting"] = line.waiting();
            if (!eng.slots.empty()) j["active"] = (int)eng.slots.size() - eng.free_slots.load();   // replies running now
            res.set_content(dump(j), "application/json; charset=utf-8"); return;
        }
        if (!eng.model) { last_info = {{"loaded", false}, {"version", TIERLM_VERSION}}; res.set_content(dump(last_info), "application/json; charset=utf-8"); return; }
        const auto& hp = eng.model->hparams();
        size_t free = 0, total = 0;
        eng.model->device_memory(free, total);
        json j = {
            {"loaded", true}, {"path", eng.path},
            {"name", eng.name.empty() ? fs::u8path(eng.path).stem().u8string() : eng.name}, {"arch", eng.model->arch()},
            {"chat_format", chat_fmt_name(eng.chat)}, {"moe", hp.is_moe()},
            {"layers", hp.n_layer}, {"experts", hp.n_expert}, {"experts_used", hp.n_expert_used},
            {"vocab", hp.n_vocab}, {"ctx", eng.slots.empty() ? eng.model->max_ctx() : eng.ctx_per}, {"n_past", eng.model->n_past()},
            {"connections", std::max<int>(1, (int)eng.slots.size())},
            {"device", eng.model->device_name()}, {"on_gpu", eng.model->on_gpu()}, {"threads", eng.threads},
            {"device_type", eng.model->integrated_gpu() ? "igpu" : eng.model->on_gpu() ? "gpu" : "cpu"},
            {"weights_bytes", eng.model->weights_bytes()}, {"kv_bytes", eng.model->kv_bytes()},
            {"device_free", free}, {"device_total", total},
            {"kv_type", eng.model->kv_type()}, {"flash_attn", eng.model->flash_attn()}, {"load_s", eng.load_s},
            {"batch", eng.model->batch()}, {"experts_default", eng.model->experts_default()}, {"seed", eng.seed},
            {"vram_weights", eng.model->placement().gpu_weights}, {"ram_weights", eng.model->placement().cpu_weights},
            {"kv_on_gpu", eng.model->placement().kv_on_gpu}, {"all_on_gpu", eng.model->placement().all_on_gpu()}, {"attn_gpu", eng.model->placement().attn_gpu},
            {"ffn_gpu", eng.model->placement().ffn_gpu}, {"kv_gpu_layers", eng.model->placement().kv_gpu_layers},
            {"split_mode", eng.model->placement().mode},
            {"gpus", [&] { json g = json::array(); for (const auto& x : eng.model->placement().gpus)
                               g.push_back({{"name", x.name}, {"weights", x.weights}, {"kv", x.kv}, {"layers", x.layers}}); return g; }()},
            {"vision", eng.vision != nullptr}, {"vision_file", eng.vision_file},
            {"think_switch", eng.think_switch}, {"thinking", eng.thinking}, {"reasoning_effort", eng.chat == ChatFmt::Harmony ? eng.effort : std::string()},
            {"vision_device", eng.vision ? eng.vision->device_name() : std::string()}, {"image_tokens", eng.image_tokens_for(0)},
            {"speculative", eng.spec != nullptr}, {"draft_max", eng.draft_max}, {"lookup", eng.lookup_len()}, {"lookup_setting", eng.lookup}, {"has_draft_block", hp.n_nextn > 0},
            {"mtp_loaded", eng.model->has_mtp()}, {"ffn_vram", eng.model->placement().ffn_vram},
            {"hot_experts", eng.model->placement().hot_experts}, {"hot_share", eng.hot_share}, {"hot", eng.model->placement().hot},
            {"live_experts", eng.model->live_hot()},
            {"ssd_weights", eng.model->placement().ssd_weights}, {"ssd_layers", eng.model->placement().ssd_layers}, {"ssd_ngram", eng.model->placement().ssd_ngram},
            {"ssd_file", eng.model->placement().ssd_file},
            {"live", [&] { const auto& s = eng.model->live_stats();
                           return json{{"picks", s.picks}, {"hits", s.hits}, {"swaps", s.swaps}, {"swap_ms", std::round(s.swap_ms)}}; }()},
            {"learned_tokens", [&] { double t = 0; for (float v : eng.usage) t += v; return hp.n_layer * hp.n_expert_used > 0 ? (long long)(t / (hp.n_layer * hp.n_expert_used)) : 0LL; }()},
            {"version", TIERLM_VERSION},
            // For "slower than expected" hints: system RAM now, and what Find the fastest settings measured here.
            {"ram_free", [&] { size_t f = 0, t = 0; if (auto d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) ggml_backend_dev_memory(d, &f, &t); return f; }()},
            {"tuned_tps", [&] {
                std::error_code ec;
                const std::string id = model_key(eng.path);
                const json& ms = settings.contains("models") ? settings["models"] : json::object();
                if (!ms.contains(id) || !ms[id].contains("tuned")) return 0.0;
                return ms[id]["tuned"].value("tps", 0.0);
            }()},
        };
        last_info = j;
        res.set_content(dump(j), "application/json; charset=utf-8");
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
        res.set_content(dump(j), "application/json; charset=utf-8");
    });

    // Models in the models folder (and, if enabled, its subfolders, as LM Studio stores them), read from
    // their headers only (fast: no weights are loaded). "id" is the path relative to the folder.
    // Every .gguf in the models folder (subfolders too if enabled), described from its header.
    // Embedding models LM Studio comes with, for the Embedding model list.
    auto scan_bundled = [&]() {
        json list = json::array();
        const auto folders = bundled_model_folders();
        for (size_t i = 0; i < folders.size(); i++) {
            std::error_code ec;
            const fs::path base = fs::u8path(folders[i]);
            auto it = fs::recursive_directory_iterator(base, fs::directory_options::skip_permission_denied, ec);
            for (; !ec && it != fs::recursive_directory_iterator() && list.size() < 50; it.increment(ec)) {
                if (it.depth() > 4) { it.disable_recursion_pending(); continue; }
                std::error_code e2;
                if (!it->is_regular_file(e2) || it->path().extension() != ".gguf") continue;
                json m = describe_model(it->path().u8string());
                if (!m.value("embedding", false)) continue;
                m["id"] = "@lmstudio/" + std::to_string(i) + "/" + it->path().lexically_relative(base).generic_u8string();
                m["builtin"] = "LM Studio";
                list.push_back(m);
            }
        }
        return list;
    };
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
        // tierlm's own folder (where Get models puts downloads), when it's another folder: always listed, subfolders too
        std::set<std::string> seen;
        for (const auto& m : list) { std::error_code e2; seen.insert(fs::weakly_canonical(fs::u8path(a.models) / fs::u8path(m.value("id", std::string())), e2).u8string()); }
        std::error_code e3;
        if (!fs::equivalent(own_models, base, e3) && fs::is_directory(own_models, e3)) {
            auto it = fs::recursive_directory_iterator(own_models, fs::directory_options::skip_permission_denied, e3);
            for (; !e3 && it != fs::recursive_directory_iterator() && list.size() < 800; it.increment(e3)) {
                if (it.depth() > 4) { it.disable_recursion_pending(); continue; }
                std::error_code e2;
                if (!it->is_regular_file(e2) || it->path().extension() != ".gguf" || gguf_is_later_part(it->path().filename().u8string())) continue;
                if (!seen.insert(fs::weakly_canonical(it->path(), e2).u8string()).second) continue;   // (the chosen folder is inside it)
                json m = describe_model(it->path().u8string());
                m["id"] = "@tierlm/" + it->path().lexically_relative(own_models).generic_u8string();
                const std::string sub = it->path().parent_path().lexically_relative(own_models).generic_u8string();
                m["folder"] = "tierlm models" + (sub == "." ? std::string() : "/" + sub);
                m["own"] = true;
                list.push_back(m);
            }
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
                  {"saved", per_model}, {"last_model", settings.value("last_model", std::string())}, {"bundled", scan_bundled()},
#ifdef _WIN32
                  {"can_pick", true},
#else
                  {"can_pick", false},
#endif
        };
        res.set_content(dump(j), "application/json; charset=utf-8");
    });

    // Change the models folder. Saved for next time.
    svr.Post("/api/folder", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) { res.status = 400; res.set_content(dump({{"error", "bad request"}}), "application/json; charset=utf-8"); return; }
        std::lock_guard<std::shared_timed_mutex> lk(mu);
        if (body.contains("recursive")) recursive = body.value("recursive", true);
        if (body.contains("folder")) {
            std::string f = body.value("folder", std::string());
            while (f.size() > 1 && (f.back() == ' ' || f.back() == '"')) f.pop_back();
            while (!f.empty() && (f.front() == ' ' || f.front() == '"')) f.erase(0, 1);
            std::error_code ec;
            if (f.empty() || !fs::is_directory(fs::u8path(f), ec)) {
                res.status = 400;
                res.set_content(dump({{"error", "folder not found: " + f}}), "application/json; charset=utf-8");
                return;
            }
            a.models = fs::absolute(fs::u8path(f), ec).make_preferred().u8string();
            settings["models_folder"] = a.models;
        }
        settings["recursive"] = recursive;
        write_settings(a.settings, settings);
        res.set_content(dump({{"ok", true}, {"folder", a.models}, {"recursive", recursive}}), "application/json; charset=utf-8");
    });

    // Native folder picker. It opens on the computer running tierlm, so only answer local requests.
    svr.Post("/api/pick-folder", [&](const httplib::Request& req, httplib::Response& res) {
#ifdef _WIN32
        if (req.remote_addr != "127.0.0.1" && req.remote_addr != "::1") {
            res.status = 403;
            res.set_content(dump({{"error", "the folder picker only works from the computer running tierlm"}}), "application/json; charset=utf-8");
            return;
        }
        bool cancelled = true;
        std::string f = pick_folder_windows(a.models, cancelled);
        res.set_content(dump({{"cancelled", cancelled}, {"folder", f}}), "application/json; charset=utf-8");
#else
        (void)req;
        res.status = 501;
        res.set_content(dump({{"error", "type the folder path instead (the picker is Windows only)"}}), "application/json; charset=utf-8");
#endif
    });

    // A model id from the page or the API, as a file inside the models folder (empty if it isn't one: only files
    // inside the models folder can be loaded).
    // A built-in embedding model of LM Studio's: "@lmstudio/<folder number>/<path inside that folder>".
    auto bundled_file = [&](const std::string& id) -> std::filesystem::path {
        const std::string pre = "@lmstudio/";
        if (id.rfind(pre, 0) != 0) return {};
        const size_t slash = id.find('/', pre.size());
        if (slash == std::string::npos) return {};
        const auto folders = bundled_model_folders();
        size_t i = 0;
        try { i = std::stoul(id.substr(pre.size(), slash - pre.size())); } catch (...) { return {}; }
        if (i >= folders.size()) return {};
        std::error_code ec;
        const auto base = std::filesystem::weakly_canonical(std::filesystem::u8path(folders[i]), ec);
        const auto file = std::filesystem::weakly_canonical(std::filesystem::u8path(folders[i]) / std::filesystem::u8path(id.substr(slash + 1)), ec);
        auto rel = file.lexically_relative(base);
        if (ec || rel.empty() || rel.begin()->string() == ".." || !std::filesystem::is_regular_file(file, ec) || file.extension() != ".gguf") return {};
        return file;
    };
    auto model_file = [&](const std::string& id_in) -> std::filesystem::path {
        std::error_code ec;
        const bool own = id_in.rfind("@tierlm/", 0) == 0;                 // in tierlm's own folder
        const std::string id = own ? id_in.substr(8) : id_in;
        const fs::path root = own ? own_models : std::filesystem::u8path(a.models);
        const auto base = std::filesystem::weakly_canonical(root, ec);
        const auto file = std::filesystem::weakly_canonical(root / std::filesystem::u8path(id), ec);
        auto rel = file.lexically_relative(base);
        if (ec || rel.empty() || rel.begin()->string() == ".." || !std::filesystem::is_regular_file(file, ec)) return {};
        return file;
    };

    svr.Post("/api/load", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("model")) {
            res.status = 400;
            res.set_content("{\"error\":\"expected {\\\"model\\\": \\\"...\\\"}\"}", "application/json; charset=utf-8");
            return;
        }
        if (!line.enter(mu, req.is_connection_closed)) {             // waits for replies already in line
            res.status = 503;
            res.set_content("{\"error\":\"busy: waited too long for earlier replies to finish\"}", "application/json; charset=utf-8");
            return;
        }
        std::lock_guard<std::shared_timed_mutex> lk(mu, std::adopt_lock);
        const auto file = model_file(body.value("model", std::string()));
        if (file.empty()) {
            res.status = 400;
            res.set_content(dump({{"error", "model not found in the models folder"}}), "application/json; charset=utf-8");
            return;
        }
        try {
            const LoadOptions o = load_options(body);
            eng.draft_max = std::min(8, std::max(0, body.value("draft_max", 0)));
            eng.lookup = std::max(-1, std::min(16, body.value("lookup", -1)));
            const int threads = body.value("threads", a.threads), seed = body.value("seed", -1);
            load(file.u8string(), o, threads, seed, body.value("hot_experts", false));
            // Remember these settings for this model, like LM Studio does.
            const std::string id = body.value("model", std::string());
            json saved = body;
            saved.erase("model");
            settings["models"][id] = saved;
            settings["last_model"] = id;
            write_settings(a.settings, settings);
            res.set_content(dump({{"ok", true}, {"device", eng.model->device_name()}, {"load_s", eng.load_s}}), "application/json; charset=utf-8");
        } catch (const std::exception& e) {
            eng.spec.reset();
            eng.look.reset();
            eng.vision.reset();
            eng.model.reset();
            eng.path.clear();
            res.status = 500;
            res.set_content(dump({{"error", e.what()}}), "application/json; charset=utf-8");
        }
    });

    // Thinking for the loaded model, as the page's Settings set it: {"thinking": false} (models that can switch it
    // off), {"effort": "low"} (gpt-oss). Applies to the page and to apps (unless a request says otherwise), and is
    // saved with the model's settings.
    svr.Post("/api/thinking", [&](const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body, nullptr, false);
        if (!body.is_object()) return oai_error(res, 400, "expected {\"thinking\": true|false} or {\"effort\": \"low\"|\"medium\"|\"high\"}");
        std::lock_guard<std::shared_timed_mutex> lk(mu);
        if (!eng.model) return oai_error(res, 400, "no model is loaded");
        std::error_code ec;
        const std::string id = model_key(eng.path);
        if (body.contains("thinking") && body["thinking"].is_boolean()) {
            eng.thinking = body["thinking"].get<bool>();
            if (settings.contains("models") && settings["models"].contains(id)) settings["models"][id]["thinking"] = eng.thinking;
        }
        if (body.contains("effort") && body["effort"].is_string()) {
            const std::string e = body["effort"].get<std::string>();
            if (e != "low" && e != "medium" && e != "high") return oai_error(res, 400, "effort is low, medium or high");
            eng.effort = e;
            if (settings.contains("models") && settings["models"].contains(id)) settings["models"][id]["effort"] = e;
        }
        write_settings(a.settings, settings);
        std::fprintf(stderr, "thinking: %s\n", eng.chat == ChatFmt::Harmony ? ("reasoning effort " + eng.effort).c_str() : eng.thinking ? "on" : "off");
        res.set_content(dump({{"ok", true}, {"thinking", eng.thinking}, {"effort", eng.effort}}), "application/json; charset=utf-8");
    });

    // ---- live context length: the cache made again at another size, without loading the model again ----
    // GET /api/context?ctx=N: what N tokens per connection would take and whether it fits where the cache is now
    // (VRAM and RAM, with room to spare), or needs a reload (more VRAM than is free: part of the model would move).
    // POST /api/context {"ctx": N}: does it, and saves N as this model's context for next time.
    auto context_plan = [&](int want) {
        const int conns = std::max<int>(1, (int)eng.slots.size());
        const int cur = eng.slots.empty() ? eng.model->max_ctx() : eng.ctx_per;
        size_t vram_tok = 0;
        const size_t all_tok = eng.model->kv_bytes_per_token(&vram_tok), ram_tok = all_tok - vram_tok;
        const long long d = (long long)(want - cur) * conns;          // positions added (or dropped), all connections
        size_t vfree = 0, vtotal = 0, rfree = 0, rtotal = 0;
        eng.model->device_memory(vfree, vtotal);
        if (auto c = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) ggml_backend_dev_memory(c, &rfree, &rtotal);
        const long long dv = d * (long long)vram_tok, dr = d * (long long)ram_tok;
        const long long spare_v = 512ll << 20, spare_r = 2048ll << 20; // (room the GPU driver and Windows keep using)
        const bool fits_v = dv <= 0 || !eng.model->on_gpu() || dv + spare_v <= (long long)vfree;
        const bool fits_r = dr <= 0 || dr + spare_r <= (long long)rfree;
        const int trained = eng.model->hparams().n_ctx;
        json j = {{"ctx", want}, {"current", cur}, {"connections", conns}, {"trained", trained},
                  {"vram_bytes", (long long)vram_tok * want * conns}, {"ram_bytes", (long long)ram_tok * want * conns},
                  {"vram_change", dv}, {"ram_change", dr}, {"vram_free", vfree}, {"ram_free", rfree},
                  {"bytes_per_token", all_tok}};
        std::string why;
        if (want < 256) why = "the context must be at least 256 tokens";
        else if (trained > 0 && want > trained) why = "longer than this model was trained for (" + std::to_string(trained) + " tokens)";
        else if (eng.model->ctx_in_use() > want * conns) why = "the conversations in the cache are longer: start a new chat first";
        j["live"] = why.empty() && fits_v && fits_r;
        j["reload"] = why.empty() && (!fits_v || !fits_r);
        if (!why.empty()) j["why"] = why;
        else if (!fits_v) j["why"] = "not enough free VRAM for the bigger cache: loading again would move part of the model to RAM";
        else if (!fits_r) j["why"] = "not enough free RAM for the bigger cache";
        return j;
    };
    svr.Get("/api/context", [&](const httplib::Request& req, httplib::Response& res) {
        std::shared_lock<std::shared_timed_mutex> lk(mu);
        if (!eng.model) return oai_error(res, 400, "no model is loaded");
        const int want = req.has_param("ctx") ? std::atoi(req.get_param_value("ctx").c_str()) : 0;
        res.set_content(dump(context_plan(want > 0 ? want : (eng.slots.empty() ? eng.model->max_ctx() : eng.ctx_per))), "application/json; charset=utf-8");
    });
    // ---- the SSD tier's drive ----
    // GET: the drives it can use with their free space, and the folder chosen ("" = each model's own drive, where the
    // file is read in place). POST {"dir": "D:\\"}: choose one (a drive's root gets a tierlm-ssd folder); used from
    // the next load. Only on the computer running tierlm (remote_refused covers POST).
    svr.Get("/api/drives", [&](const httplib::Request& req, httplib::Response& res) {
        if (!is_local(req)) return oai_error(res, 403, "only on the computer running tierlm");
        json list = json::array();
        for (const DriveInfo& d : list_drives())
            list.push_back({{"path", d.path}, {"label", d.label}, {"free", d.free_bytes}, {"total", d.total_bytes}});
        std::string dir = settings.value("ssd_dir", std::string());
        unsigned long long fr = 0, tot = 0;
        const bool known = !dir.empty() && drive_space(fs::u8path(dir).root_path().u8string(), fr, tot);
        res.set_content(dump(json{{"drives", list}, {"ssd_dir", dir}, {"ssd_free", known ? fr : 0ull}}), "application/json; charset=utf-8");
    });
    svr.Post("/api/ssd", [&](const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body, nullptr, false);
        if (!body.is_object()) return oai_error(res, 400, "expected {\"dir\": ...}");
        std::string dir = body.value("dir", std::string());
        if (!dir.empty()) {
            fs::path p = fs::u8path(dir);
            if (p == p.root_path()) p /= "tierlm-ssd";             // a whole drive: our own folder on it
            std::error_code ec;
            fs::create_directories(p, ec);
            if (!fs::is_directory(p, ec)) return oai_error(res, 400, "can't use " + p.u8string() + ": " + (ec ? ec.message() : "not a folder"));
            dir = p.u8string();
        }
        settings["ssd_dir"] = dir;
        write_settings(a.settings, settings);
        std::fprintf(stderr, "SSD tier: %s (from the next load)\n", dir.empty() ? "each model's own drive" : dir.c_str());
        res.set_content(dump(json{{"ok", true}, {"ssd_dir", dir}}), "application/json; charset=utf-8");
    });

    svr.Post("/api/context", [&](const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body, nullptr, false);
        const int want = body.is_object() ? body.value("ctx", 0) : 0;
        if (!line.enter(mu, req.is_connection_closed)) return oai_error(res, 503, "busy: waited too long for earlier replies to finish");
        std::lock_guard<std::shared_timed_mutex> lk(mu, std::adopt_lock);
        if (!eng.model) return oai_error(res, 400, "no model is loaded");
        const json plan = context_plan(want);
        if (!plan.value("live", false)) return oai_error(res, 409, plan.value("why", std::string("can't change the context now")));
        const int conns = std::max<int>(1, (int)eng.slots.size());
        const auto t0 = std::chrono::steady_clock::now();
        try { eng.model->resize_ctx(want * conns); }
        catch (const std::exception& e) { return oai_error(res, 409, e.what()); }
        if (!eng.slots.empty()) eng.ctx_per = want;
        // (kept for next time with the other load settings of this model)
        std::error_code ec;
        const std::string id = model_key(eng.path);                  // (as loaded_id below)
        if (settings.contains("models") && settings["models"].contains(id) && settings["models"][id].is_object()) {
            settings["models"][id]["ctx"] = want;
            write_settings(a.settings, settings);
        }
        const double s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "context: %d tokens%s, changed in %.2f s without loading again\n", want,
                     conns > 1 ? (" per connection (" + std::to_string(conns) + " connections)").c_str() : "", s_);
        res.set_content(dump({{"ok", true}, {"ctx", want}, {"seconds", s_}}), "application/json; charset=utf-8");
    });

    // The embedding model's slot (see "The embedding model" below).
    struct EmbedSlot {
        std::mutex mu;                       // guards the slot; held while embedding
        std::unique_ptr<Encoder> enc;
        Tokenizer tok;
        std::string id, path, name;
        double load_s = 0;
    } emb;
    // Eject: frees the model and everything that belongs to it (its vision file, draft block, the replies'
    // caches), giving its VRAM and RAM back; {"embedding": true} frees the embedding model too (it stays the
    // saved choice, for the next start). Waits for running replies to finish.
    svr.Post("/api/unload", [&](const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body.empty() ? "{}" : req.body, nullptr, false);
        const bool with_embed = body.is_object() && body.value("embedding", false);
        std::lock_guard<std::shared_timed_mutex> lk(mu);
        const bool had = eng.model != nullptr;
        eng.save_usage(true);
        eng.batch.finish();
        eng.slots.clear(); eng.free_slots = 0;
        eng.spec.reset();
        eng.look.reset();
        eng.vision.reset(); eng.vision_file.clear();
        eng.model.reset();
        eng.cached.clear();
        eng.path.clear();
        bool emb_freed = false;
        if (with_embed) { std::lock_guard<std::mutex> le(emb.mu); emb_freed = emb.enc != nullptr; emb.enc.reset(); emb.id.clear(); emb.path.clear(); emb.name.clear(); }
        if (had || emb_freed)
            std::fprintf(stderr, "ejected: %s%s; %s memory is free again\n", had ? "the model" : "",
                         emb_freed ? (had ? " and the embedding model" : "the embedding model") : "", had && emb_freed ? "their" : "its");
        res.set_content("{\"ok\":true}", "application/json; charset=utf-8");
    });

    svr.Post("/api/reset", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::shared_timed_mutex> lk(mu);
        eng.save_current();                                      // a new chat: the old one may be opened again
        if (eng.model) eng.model->reset();
        eng.cached.clear();
        res.set_content("{\"ok\":true}", "application/json; charset=utf-8");
    });

    // One reply, shared by the web page and the OpenAI-compatible API. The caller holds `mu`.
    //   on_prompt  once, after the conversation has been read
    //   on_token   for each piece of new text (with the experts each layer picked); return false to stop
    // `stops` are strings that end the reply (OpenAI's "stop"); text that might be the start of one is held back.
    struct Gen { size_t n_prompt = 0, reused = 0, n_gen = 0, drafted = 0, accepted = 0, checks = 0, looked = 0, look_ok = 0; double prompt_s = 0, gen_s = 0, guess_ms = 0, check_ms = 0, catch_ms = 0;
                 std::string stop = "length", text, full; };
    // The tool calls in a reply, in the model's own format.
    auto parse_calls = [&](const std::string& full, const agent::ojson& tools) {
        return eng.chat == ChatFmt::Harmony ? harmony::parse(full, tools) : agent::parse_tool_calls(full, tools);
    };
    // With `tools`, the model is told about them, and a tool call it writes is kept out of the streamed text (it is
    // in `full`, the whole reply as the model wrote it, for agent::parse_tool_calls).
    // The cache length of the reply being written on this thread (several connections: each has its own).
    static thread_local int tl_past = 0;
    static thread_local int tl_image_tokens = 0;      // this request's image detail (0: automatic)
    static thread_local int tl_think = -1;            // this request's thinking: 1 on, 0 off, -1 the model's setting
    // gpt-oss: this request's reasoning effort, 1 low, 2 medium, 3 high, 0 the model's setting (a number: a thread-local
    // std::string used from the route lambdas isn't set up on their threads with GCC)
    static thread_local int tl_effort = 0;
    // How the assistant turn starts (format_chat's `think`), for the loaded model and this request.
    auto think_mode = [&]() -> int {
        if (!eng.think_switch) return eng.think_starts ? 1 : -1;  // (DeepSeek R1: always thinks, its template opens the block)
        const bool on = tl_think >= 0 ? tl_think == 1 : eng.thinking;
        return on ? (eng.think_starts ? 1 : -1) : 0;
    };
    auto effort_now = [&]() { return tl_effort == 1 ? std::string("low") : tl_effort == 2 ? std::string("medium") : tl_effort == 3 ? std::string("high") : eng.effort; };
    // A request's own choice, in the forms apps send it: {"chat_template_kwargs": {"enable_thinking": false}} (vLLM,
    // llama.cpp), "enable_thinking", "think" (Ollama), "thinking" (tierlm's page); "reasoning_effort" for gpt-oss
    // ("none" or "minimal" turn thinking off for the others).
    auto take_think = [&](const json& body) {
        tl_think = -1; tl_effort = 0;
        if (!body.is_object()) return;
        auto flag = [&](const json& j, const char* k) { if (j.is_object() && j.contains(k) && j[k].is_boolean()) tl_think = j[k].get<bool>() ? 1 : 0; };
        flag(body, "thinking"); flag(body, "think"); flag(body, "enable_thinking");
        if (body.contains("chat_template_kwargs")) flag(body["chat_template_kwargs"], "enable_thinking");
        if (body.contains("reasoning_effort") && body["reasoning_effort"].is_string()) {
            const std::string e = body["reasoning_effort"].get<std::string>();
            tl_effort = e == "low" ? 1 : e == "medium" ? 2 : e == "high" ? 3 : 0;
            if (e == "none" || e == "minimal") tl_think = 0; else if (tl_think < 0 && !e.empty()) tl_think = 1;
        }
    };
    auto generate = [&](const json& messages, const Sampling& sp, int max_tokens, const std::vector<std::string>& stops_in,
                        const std::function<void(const json&)>& on_prompt,
                        const std::function<bool(const std::string&, double, const std::vector<int32_t>&)>& on_token,
                        const agent::ojson& tools = agent::ojson()) {
        Gen r;
        // harmony (gpt-oss): a reply is several messages (reasoning, then the answer or a tool call) separated by
        // special tokens; it ends only at <|return|> or <|call|>, and is shown as <think>reasoning</think>answer.
        const bool harm = eng.chat == ChatFmt::Harmony;
        const bool use_tools = tools.is_array() && !tools.empty();
        std::vector<std::string> stops = stops_in;
        if (use_tools && !harm) stops.push_back("<tool_response>");     // the model must not write the tool's answer itself
        if (max_tokens <= 0) max_tokens = INT_MAX;    // no limit: until the model finishes or the context is full
        auto& model = eng.model; auto& tok = eng.tok;
        const auto& hp = model->hparams();
        const int L = hp.n_layer, E = hp.n_expert, K = hp.n_expert_used;

        json msgs = messages;
        const std::vector<std::string> pics = take_pictures(msgs);
        if (!pics.empty() && harm) throw std::runtime_error("this model can't read images");
        const int think = think_mode();
        // (the template opened the <think> block: the reply is shown, split and saved with it, as the model wrote it)
        const std::string pre = think == 1 ? "<think>\n" : "";
        std::vector<int32_t> toks = tok.encode(harm ? harmony::prompt(msgs, tools, eng.identity, effort_now())
                                                    : format_chat(with_tools(msgs, tools, eng.tool_fmt), eng.chat, tok, think), true);
        if (!pics.empty()) {                                      // each image's <|image_pad|> becomes its words
            const int32_t pad = tok.find("<|image_pad|>");
            std::vector<int32_t> with;
            size_t k = 0;
            for (int32_t t : toks) {
                if (t != pad || pad < 0) { with.push_back(t); continue; }
                if (k >= pics.size()) throw std::runtime_error("the conversation has more image places than images");
                const Engine::Pic p = eng.image(pics[k++], eng.threads, tl_image_tokens);
                for (int i = 0; i < p.n; ++i) with.push_back(p.first + i);
            }
            if (k != pics.size()) throw std::runtime_error("this model's chat format has no place for images");
            toks.swap(with);
        }
        const bool has_pics = !pics.empty() || std::any_of(toks.begin(), toks.end(), Model::is_image_id);
        // Several connections: this reply gets a slot (a conversation of the model), the free one whose cache
        // already holds the longest start of this conversation. Its steps go through the batcher.
        const bool par = !eng.slots.empty();
        int seq = -1;
        struct SlotGuard {
            Engine& e; int s;
            ~SlotGuard() {
                if (s < 0) return;
                { std::lock_guard<std::mutex> lk(e.batch.m); --e.batch.active; }
                e.batch.cv.notify_all();
                { std::lock_guard<std::mutex> lk(e.slots_mu); e.slots[s].busy = false; }
            }
        } slot_guard{eng, -1};
        if (par) {
            std::lock_guard<std::mutex> lk(eng.slots_mu);
            size_t best_n = 0;
            for (int s = 0; s < (int)eng.slots.size(); ++s) {
                if (eng.slots[s].busy) continue;
                const auto& c = eng.slots[s].cached;
                size_t n = 0;
                while (n < c.size() && n < toks.size() && c[n] == toks[n]) ++n;
                if (hp.hybrid && n < c.size()) n = 0;                    // (hybrid models can't rewind)
                if (seq < 0 || n > best_n) { seq = s; best_n = n; }
            }
            if (seq < 0) throw std::runtime_error("no free connection");   // (the queue lets a reply in only when one is free)
            eng.slots[seq].busy = true;
            slot_guard.s = seq;
            std::lock_guard<std::mutex> lb(eng.batch.m);
            ++eng.batch.active;
        }
        auto& cached = par ? eng.slots[seq].cached : eng.cached;
        auto& rng = par ? eng.slots[seq].rng : eng.rng;
        if (eng.seed >= 0) rng.seed((uint32_t)eng.seed);    // same seed, same reply
        const int ctx_lim = par ? eng.ctx_per : model->max_ctx();
        auto npast = [&] { return par ? model->seq_past(seq) : model->n_past(); };
        auto evalm = [&](const std::vector<int32_t>& ids, std::vector<int32_t>* rt) {
            return par ? eng.batch.eval(seq, ids, rt) : model->eval(ids, eng.threads, rt);
        };
        if ((int)toks.size() + 1 >= ctx_lim) throw std::runtime_error("conversation is longer than the context window; start a new chat");

        // Reuse whatever part of the KV cache still matches this conversation.
        auto prefix = [&](const std::vector<int32_t>& a) {
            size_t n = 0;
            while (n < a.size() && n < toks.size() && a[n] == toks[n]) ++n;
            return n;
        };
        size_t common = prefix(cached);
        // The last reply comes back as text, and text can split into tokens differently from how the model wrote
        // it. When the conversation's text still continues everything in the cache, keep the model's own tokens
        // (it's the same text) so nothing has to be read again; hybrid models couldn't go back that far at all.
        if (!par && !has_pics && common < cached.size() && cached.size() < toks.size()) {
            try {
                const std::string a = tok.decode(std::vector<int32_t>(cached.begin() + common, cached.end()), false);
                const std::string b = tok.decode(std::vector<int32_t>(toks.begin() + common, toks.end()), false);
                if (!a.empty() && b.size() > a.size() && b.compare(0, a.size(), a) == 0) {
                    std::vector<int32_t> joined(cached.begin(), cached.end());
                    const std::vector<int32_t> rest = tok.encode(b.substr(a.size()), true);
                    joined.insert(joined.end(), rest.begin(), rest.end());
                    if (tok.decode(joined, false) == tok.decode(toks, false)) { toks.swap(joined); common = cached.size(); }
                }
                if (std::getenv("TIERLM_TRACE_CACHE")) {
                    size_t d = 0;
                    while (d < a.size() && d < b.size() && a[d] == b[d]) ++d;
                    std::fprintf(stderr, "cache trace: tokens differ at %zu of %zu cached; text %s; first difference at char %zu:\n  cache: [%s]\n  now:   [%s]\n",
                                 prefix(cached), cached.size(), common == cached.size() ? "continues the cache" : "differs",
                                 d, a.substr(d > 20 ? d - 20 : 0, 60).c_str(), b.substr(d > 20 ? d - 20 : 0, 60).c_str());
                }
            } catch (const std::exception&) {}
        }
        if (std::getenv("TIERLM_TRACE_CACHE"))
            std::fprintf(stderr, "cache trace: reuse %zu, cached %zu, n_past %d, checkpoint %d\n", common, cached.size(), npast(), model->checkpoint_pos());
        if (!par) {
            // A saved conversation that this request continues further than the cache does: swap it in (saving the
            // cache's own conversation first). Hybrid models can't rewind, so for them it must continue all of it.
            auto best = eng.saved.end();
            size_t best_n = common;
            for (auto it = eng.saved.begin(); it != eng.saved.end(); ++it) {
                size_t n = prefix(it->tokens);
                if (hp.hybrid && n < it->tokens.size()) n = 0;
                if (n > best_n + 64) { best = it; best_n = n; }
            }
            if (best != eng.saved.end()) {
                auto c = std::move(*best);
                eng.saved.erase(best);
                eng.save_current();
                const auto t0 = Clock::now();
                model->restore(c.snap);
                if (eng.spec) eng.spec->set_state(c.has_spec ? c.spec : Speculator::State{});
                cached = c.tokens;
                common = prefix(cached);
                std::fprintf(stderr, "cache: brought back a saved %zu-token conversation in %.2f s\n", cached.size(), since(t0));
                eng.saved.push_front(std::move(c));                    // (kept: the conversation may branch here again)
            } else if (cached.size() >= common + 256 &&
                       !(hp.hybrid && model->checkpoint_pos() >= 0 && common >= (size_t)model->checkpoint_pos())) {
                eng.save_current();                                      // about to be overwritten: keep it
                                                                         // (not when a hybrid model just goes back to its checkpoint)
            }
        }
        if (common == toks.size()) common = toks.size() - 1;      // must evaluate at least one token
        if (par) model->seq_truncate(seq, (int)common); else model->truncate((int)common);
        common = (size_t)npast();             // hybrid models start over rather than rewind
        cached.resize(common);

        std::vector<int32_t> suffix(toks.begin() + common, toks.end()), routing;
        auto p0 = Clock::now();
        std::vector<float> lg;
        // (speculation: one conversation only, and not with images: the draft block reads text only)
        Speculator* const spx = par || has_pics ? nullptr : eng.active_spec();
        // Hybrid models can't rewind, so they keep one point to go back to: the start of this reply's own turn. The
        // next message repeats everything before it word for word (the turn itself is written differently in the
        // history: no empty thinking block), so it carries on from there instead of reading the conversation again.
        size_t split = 0;
        if (!par && hp.hybrid) {
            static const char* const marks[] = {"<|im_start|>", "<start_of_turn>", "<|start_header_id|>", "<|start|>"};
            int32_t mark = -1;
            for (const char* m : marks) if ((mark = tok.find(m)) >= 0) break;
            if (mark >= 0) for (size_t i = toks.size(); i-- > common + 1;) if (toks[i] == mark) { split = i; break; }
        }
        if (split > common) {
            const std::vector<int32_t> head(toks.begin(), toks.begin() + split);
            std::vector<int32_t> rest_routing;
            if (spx) spx->prompt(head, eng.threads, &routing);
            else evalm(std::vector<int32_t>(head.begin() + common, head.end()), &routing);
            model->checkpoint();
            if (spx) spx->mark();
            if (spx) { lg = spx->prompt(toks, eng.threads, &rest_routing); spx->reset_stats(); }
            else lg = evalm(std::vector<int32_t>(toks.begin() + split, toks.end()), &rest_routing);
            routing.insert(routing.end(), rest_routing.begin(), rest_routing.end());
        } else {
            if (spx) { lg = spx->prompt(toks, eng.threads, &routing); spx->reset_stats(); }   // also feeds the draft block
            else lg = evalm(suffix, &routing);
            if (!par && hp.hybrid && common == 0) { model->checkpoint(); if (spx) spx->mark(); }
        }
        r.prompt_s = since(p0);
        std::fprintf(stderr, "chat: %zu prompt tokens, %zu reused from the cache, %zu read at %.0f tok/s\n", toks.size(), (size_t)common,
                     toks.size() - common, (toks.size() - common) / std::max(r.prompt_s, 1e-9));
        r.n_prompt = toks.size();
        r.reused = common;
        cached = toks;

        std::vector<int> counts((size_t)L * E, 0);                // experts used while reading the prompt
        if (K > 0) for (size_t i = 0; i < routing.size(); ++i) if (routing[i] >= 0) counts[((i / K) % L) * E + routing[i]]++;   // (-1: a layer without experts)
        eng.add_usage(routing, K);
        if (on_prompt)
            on_prompt({{"n_prompt", toks.size()}, {"n_new", suffix.size()}, {"reused", common}, {"prompt_s", r.prompt_s},
                       {"prompt_tps", suffix.size() / std::max(r.prompt_s, 1e-9)}, {"counts", counts}, {"n_past", npast()}});

        std::vector<int32_t> out;
        size_t shown = 0;
        auto g0 = Clock::now();
        auto emit = [&](const std::string& visible, size_t upto) {
            if (upto <= shown) return true;
            const std::string delta = visible.substr(shown, upto - shown);
            shown = upto;
            r.text += delta;
            tl_past = npast();
            return on_token ? on_token(delta, out.size() / std::max(since(g0), 1e-9), routing) : true;
        };
        std::string visible, raw;                                 // raw: the reply as the model wrote it (harmony: before display)
        const int32_t h_return = tok.find("<|return|>"), h_call = tok.find("<|call|>"), h_end = tok.find("<|end|>"), h_eot = tok.find("<|endoftext|>");
        auto is_end = [&](int32_t t) {
            if (harm) return t == h_return || t == h_call || t == h_eot || (t == tok.eos() && t != h_end);
            return t == tok.eos() || t == tok.eot() || tok.is_control(t);
        };
        // Recent tokens for the repetition penalty (the cache, then `extra`), leaving out control tokens such as the
        // end-of-reply marker: penalising those would stop the model from finishing.
        auto recent_with = [&](const std::vector<int32_t>& extra) {
            std::vector<int32_t> rc;
            auto skip = [&](int32_t t) { return is_end(t) || tok.is_control(t); };
            for (size_t j = extra.size(); j > 0 && (int)rc.size() < sp.repeat_last_n; --j) if (!skip(extra[j - 1])) rc.push_back(extra[j - 1]);
            for (size_t j = cached.size(); j > 0 && (int)rc.size() < sp.repeat_last_n; --j) if (!skip(cached[j - 1])) rc.push_back(cached[j - 1]);
            return rc;
        };
        // Adds a chosen token to the reply and streams its text. False when the reply must end here.
        auto add = [&](int32_t id) {
            out.push_back(id);
            const std::string text = tok.decode(out);
            raw = pre + text.substr(0, utf8_complete(text));
            visible = harm ? harmony::display(raw) : raw;
            // A stop string: end the reply just before it.
            size_t cut = std::string::npos;
            for (const auto& st : stops)
                if (!st.empty()) { size_t p = visible.find(st, shown > st.size() ? shown - st.size() : 0); if (p < cut) cut = p; }
            if (cut != std::string::npos) {
                r.full = harm ? raw : visible.substr(0, cut);
                const size_t tc = use_tools && !harm ? agent::tool_call_pos(r.full) : std::string::npos;
                emit(visible, std::max(std::min(cut, tc), shown));
                r.stop = "stop";
                return false;
            }
            // Hold back text that could be the start of a stop string.
            size_t hold = 0;
            for (const auto& st : stops)
                for (size_t k = st.empty() ? 0 : std::min(st.size() - 1, visible.size()); k > hold; --k)
                    if (visible.compare(visible.size() - k, k, st, 0, k) == 0) { hold = k; break; }
            size_t limit = visible.size() - hold;
            if (use_tools && !harm) {                              // a tool call isn't shown as text
                const size_t tc = agent::tool_call_pos(visible);
                if (tc != std::string::npos) limit = std::min(limit, tc);
                else if (agent::answer_start(visible) != std::string::npos) {
                    const std::string mk = agent::kToolOpen;
                    for (size_t k = std::min(mk.size() - 1, visible.size()); k > 0; --k)
                        if (visible.compare(visible.size() - k, k, mk, 0, k) == 0) { limit = std::min(limit, visible.size() - k); break; }
                }
            }
            if (!emit(visible, limit)) { r.stop = "abort"; return false; }
            return true;
        };
        if (!spx) {
            for (;;) {
                if ((int)out.size() >= max_tokens) { r.stop = "length"; break; }
                const int32_t id = sample(lg.data(), lg.size(), sp, recent_with({}), rng);
                if (is_end(id)) { r.stop = "end"; break; }
                if (npast() + 1 >= ctx_lim) { r.stop = "context"; break; }
                lg = evalm({id}, &routing);                               // also tells us which experts it used
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
                return sample(row, (size_t)hp.n_vocab, sp, recent_with(extra), rng);
            };
            // Sampling: speculative sampling (guesses drawn from the draft block's own probabilities, kept by the
            // ratio of the model's to the draft block's); the reply follows the model's distribution exactly either
            // way, but more guesses are kept (Flash-Next IQ3_XXS, 5090, a story at 0.7: 63 tok/s against 46-52).
            // TIERLM_SPEC_SAMPLING=0: guess the best, keep it if the model's draw agrees.
            static const bool spec_sampling = !(std::getenv("TIERLM_SPEC_SAMPLING") && !std::strcmp(std::getenv("TIERLM_SPEC_SAMPLING"), "0"));
            struct ClearSampling { Speculator* s; ~ClearSampling() { s->clear_sampling(); } } clear_sampling{spx};
            if (sp.temperature > 0.0f && spec_sampling)
                spx->set_sampling([&](const float* row, const std::vector<int32_t>& ctx) {
                    return distribution(row, (size_t)hp.n_vocab, sp, recent_with(ctx));
                }, &rng);
            int32_t x = pick(lg.data(), {});
            for (;;) {
                if ((int)out.size() >= max_tokens) { r.stop = "length"; break; }
                if (is_end(x)) { r.stop = "end"; break; }
                if (model->n_past() + 1 >= model->max_ctx()) { r.stop = "context"; break; }
                cur = x;
                const std::vector<int32_t> prod = spx->step(x, eng.threads, pick, is_end, K > 0 ? &kept : nullptr);
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
            const SpecStats& st = spx->stats();
            r.drafted = st.drafted; r.accepted = st.accepted; r.checks = st.steps; r.looked = st.looked; r.look_ok = st.look_ok;
            r.guess_ms = st.guess_s * 1e3; r.check_ms = st.check_s * 1e3; r.catch_ms = st.catch_s * 1e3;
        }
        if (r.stop != "stop") r.full = harm ? raw : visible;
        if (r.stop != "stop" && r.stop != "abort") {                                  // release held-back text
            const size_t tc = use_tools && !harm ? agent::tool_call_pos(visible) : std::string::npos;
            emit(visible, tc == std::string::npos ? visible.size() : tc);
        }
        r.n_gen = out.size();
        r.gen_s = since(g0);
        eng.save_usage();
        std::string vram_note;                                   // how much of this reply's expert work VRAM did
        {
            std::lock_guard<std::mutex> lk(eng.usage_mu);
            if (eng.picks > 0) vram_note = "; VRAM experts did " + std::to_string((int)std::lround(100.0 * eng.picks_vram / eng.picks)) + "% of this reply's expert work";
            eng.picks = eng.picks_vram = 0;
        }
        if (model->live_hot()) {                                 // the VRAM set moves: count where each pick really was
            const Model::LiveStats ls = model->live_stats();
            if (ls.picks > 0) {
                char b[160];
                std::snprintf(b, sizeof b, "; VRAM experts did %d%% of this reply's expert work (%lld moved into VRAM, %.0f ms)",
                              (int)std::lround(100.0 * ls.hits / ls.picks), ls.swaps, ls.swap_ms);
                vram_note = b;
            }
            model->reset_live_stats();
        }
        if (r.checks > 0 && r.drafted > 0)
            std::fprintf(stderr, "reply: %zu tokens at %.1f tok/s (%s); speculative: %.1f tokens per check, %zu of %zu guesses kept"
                         " (prompt lookup: %zu of %zu); per check: guessing %.1f ms, checking %.1f ms, draft block catching up %.1f ms\n",
                         r.n_gen, r.n_gen / std::max(r.gen_s, 1e-9), r.stop.c_str(),
                         (double)r.n_gen / r.checks, r.accepted, r.drafted, r.look_ok, r.looked,
                         r.guess_ms / r.checks, r.check_ms / r.checks, r.catch_ms / r.checks);
        else
            std::fprintf(stderr, "reply: %zu tokens at %.1f tok/s (%s)\n", r.n_gen, r.n_gen / std::max(r.gen_s, 1e-9), r.stop.c_str());
        if (!vram_note.empty()) std::fprintf(stderr, "reply: %s\n", vram_note.c_str() + 2);
        return r;
    };

    // ---- long conversations: fitting the context, the archive and looking back at it ----
    // Saved chats live in a "chats" folder next to the settings file (see chats.h).
    const fs::path chats_dir = [&] { std::error_code ec; return fs::absolute(fs::u8path(a.settings), ec).parent_path() / "chats"; }();
    // Set once the embedding slot exists (further down): text -> unit vector, or empty when no embedding model is loaded.
    std::function<std::vector<float>(const std::string&, bool)> embed_for_recall;
    // How many tokens a conversation's prompt is, exactly as generate() will build it.
    auto prompt_tokens = [&](const json& messages, const agent::ojson& tools) {
        const bool harm = eng.chat == ChatFmt::Harmony;
        json msgs = messages;
        std::vector<std::string> pics;
        try { pics = take_pictures(msgs); } catch (...) {}
        int n = (int)eng.tok.encode(harm ? harmony::prompt(msgs, tools, eng.identity, effort_now())
                                         : format_chat(with_tools(msgs, tools, eng.tool_fmt), eng.chat, eng.tok, think_mode()), true).size();
        for (const std::string& b : pics) {                        // each image: its words instead of one pad token
            int w = 0, h = 0;
            if (eng.vision && image_size((const uint8_t*)b.data(), b.size(), &w, &h)) n += eng.vision->tokens_for(w, h, eng.image_tokens_for(tl_image_tokens)) - 1;
        }
        return n;
    };
    // Drops the oldest messages (never the system message or the last one) until the prompt is at most `budget`
    // tokens, and so that what's left starts with a message from the user. Returns how many were dropped.
    auto drop_oldest = [&](json& messages, const agent::ojson& tools, int budget) {
        const size_t first = !messages.empty() && messages[0].value("role", std::string()) == "system" ? 1 : 0;
        if (messages.size() <= first + 1 || prompt_tokens(messages, tools) <= budget) return 0;
        auto without = [&](size_t k) {                          // the first k messages after the system one left out
            json m = json::array();
            for (size_t i = 0; i < messages.size(); ++i) if (i < first || i >= first + k) m.push_back(messages[i]);
            return m;
        };
        size_t lo = 1, hi = messages.size() - first - 1;        // binary search for the fewest that's enough
        while (lo < hi) {
            const size_t mid = (lo + hi) / 2;
            if (prompt_tokens(without(mid), tools) <= budget) hi = mid; else lo = mid + 1;
        }
        size_t k = lo;
        while (first + k < messages.size() - 1 && messages[first + k].value("role", std::string()) != "user") ++k;
        messages = without(k);
        return (int)k;
    };

    svr.Post("/api/chat", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content("{\"error\":\"expected {\\\"messages\\\": [...]}\"}", "application/json; charset=utf-8");
            return;
        }
        auto lock = std::make_shared<Hold>();                              // held until the stream ends
        res.set_header("Cache-Control", "no-cache");
        const bool local = is_local(req);                                   // saved chats are read only for this computer

        res.set_chunked_content_provider("text/event-stream", [&, body, lock, local](size_t, httplib::DataSink& sink) {
            auto send = [&](const json& j) { std::string s = "data: " + dump(j) + "\n\n"; return sink.write(s.data(), s.size()); };
            // Wait in line, telling the page how many requests are ahead ("queue" events), so it can say so.
            bool gone = false;
            if (!line.enter_reply(mu, *lock, reply_slots, [&] { return gone || !sink.is_writable(); },
                                  [&](int ahead) { if (!send({{"type", "queue"}, {"ahead", ahead}})) gone = true; })) {
                if (!gone) send({{"type", "error"}, {"message", "waited too long for earlier replies to finish"}});
                sink.done();
                return true;
            }
            if (!eng.model) {
                send({{"type", "error"}, {"message", "no model loaded: choose one on the setup screen"}});
                sink.done();
                return true;
            }
            try {
                Sampling sp;
                sp.temperature = body.value("temperature", sp.temperature);
                sp.top_p = body.value("top_p", sp.top_p);
                sp.top_k = body.value("top_k", sp.top_k);
                sp.min_p = body.value("min_p", sp.min_p);
                sp.repeat_penalty = body.value("repeat_penalty", sp.repeat_penalty);
                sp.presence_penalty = body.value("presence_penalty", sp.presence_penalty);
                sp.frequency_penalty = body.value("frequency_penalty", sp.frequency_penalty);
                sp.repeat_last_n = body.value("repeat_last_n", sp.repeat_last_n);
                // Longest reply in tokens; 0 or less (the default) means no limit: the reply ends when the model
                // finishes or the context window is full. Thinking models (Qwen3.x) need thousands of tokens.
                const int max_tokens = body.value("max_tokens", -1);
                tl_image_tokens = body.value("image_tokens", 0);      // image detail (0: automatic)
                take_think(body);
                const int L = eng.model->hparams().n_layer, K = eng.model->hparams().n_expert_used;
                // The workspace (tierlm's own page): the model gets the folder's tools and is told about them. The
                // tools only describe; running them is /api/workspace/tool, which only this computer's page can use.
                json messages = body["messages"];
                agent::ojson tools = agent::ojson::array();
                const std::string ws = settings.value("workspace_folder", std::string());
                if (body.value("workspace", false) && !ws.empty()) {
                    tools = workspace::tools();
                    const std::string note = workspace::instructions(ws);
                    if (!messages.empty() && messages[0].value("role", std::string()) == "system")
                        messages[0]["content"] = messages[0].value("content", std::string()) + "\n\n" + note;
                    else messages.insert(messages.begin(), json{{"role", "system"}, {"content", note}});
                }
                // A saved chat with an archive: its summary goes in the system message (it rarely changes, so the
                // cache keeps it), and the archived passages that best match the new message go just before it.
                const std::string chat_id = local ? body.value("chat_id", std::string()) : std::string();
                if (!chat_id.empty() && !messages.empty() && messages.back().value("role", std::string()) == "user" &&
                    messages.back()["content"].is_string()) {
                    json c;
                    try { c = chats::load(chats_dir, chat_id); } catch (...) {}
                    if (c.is_object()) {
                        const std::string summary = c.value("summary", std::string());
                        bool any_archived = false;
                        for (const auto& m : c.value("messages", json::array())) if (m.value("archived", false)) { any_archived = true; break; }
                        if (any_archived) {
                            const std::string note = "Earlier parts of this conversation were archived to save memory: you no longer see them, "
                                "but relevant passages from them are shown to you with a new message when there are any." +
                                (summary.empty() ? std::string() : "\n\nSummary of the archived part: " + summary);
                            if (messages[0].value("role", std::string()) == "system")
                                messages[0]["content"] = messages[0].value("content", std::string()) + "\n\n" + note;
                            else messages.insert(messages.begin(), json{{"role", "system"}, {"content", note}});
                            const std::string q = messages.back()["content"].get<std::string>();
                            std::vector<chats::Passage> ps;
                            try { ps = chats::recall(c, q, 6000, 4, embed_for_recall); } catch (...) {}
                            if (!ps.empty()) {
                                std::string ctxt = "[From the archived part of this conversation, possibly relevant:]\n";
                                json items = json::array();
                                for (const auto& p : ps) {
                                    const std::string who = p.role == "user" ? "the user" : p.role == "assistant" ? "you" : "a tool";
                                    ctxt += "\n--- earlier, from " + who + (p.time ? " at " + chats::local_time(p.time, "%Y-%m-%d %H:%M") : "") + ":\n" + p.text + "\n";
                                    items.push_back({{"msg", p.msg}, {"role", p.role}, {"time", p.time}, {"text", p.text.substr(0, 300)}});
                                }
                                ctxt += "\n[End of the archived passages. The new message:]\n\n";
                                messages.back()["content"] = ctxt + messages.back()["content"].get<std::string>();
                                send({{"type", "recall"}, {"items", items}});
                            }
                        }
                    }
                }
                // Too long for the context: with "overflow": "archive", the oldest messages are left out (down to
                // about 60% of the context, so it isn't needed again at once, which would re-read the prompt every
                // time) and the page is told how many, to archive them; otherwise the page offers what to do.
                {
                    const int ctx = eng.slots.empty() ? eng.model->max_ctx() : eng.ctx_per;
                    const int reserve = max_tokens > 0 ? std::min(max_tokens, ctx / 2) : std::max(256, ctx / 8);
                    const int need = prompt_tokens(messages, tools);
                    if (body.value("overflow", std::string()) == "archive" && need > ctx - reserve) {
                        const int n = drop_oldest(messages, tools, std::min(ctx * 6 / 10, ctx - reserve));
                        if (n > 0) send({{"type", "archived"}, {"count", n}});
                    } else if (need + 1 >= ctx) {
                        send({{"type", "error"}, {"code", "context_full"}, {"tokens", need}, {"ctx", ctx},
                              {"message", "the conversation is longer than the context window"}});
                        sink.done();
                        return true;
                    }
                }
                Gen g = generate(messages, sp, max_tokens, {},
                    [&](const json& p) { json j = p; j["type"] = "prompt"; send(j); },
                    [&](const std::string& delta, double tps, const std::vector<int32_t>& routing) {
                        json picks = json::array();
                        if (K > 0 && routing.size() >= (size_t)L * K)
                            for (int l = 0; l < L; ++l)
                                picks.push_back(std::vector<int32_t>(routing.begin() + (size_t)l * K, routing.begin() + (size_t)(l + 1) * K));
                        return send({{"type", "token"}, {"text", delta}, {"tps", tps}, {"n_past", tl_past}, {"routing", picks}});
                    }, tools);
                json done = {{"type", "done"}, {"n_gen", g.n_gen}, {"gen_s", g.gen_s},
                             {"gen_tps", g.n_gen / std::max(g.gen_s, 1e-9)}, {"stop", g.stop}, {"n_past", tl_past}};
                if (!tools.empty()) {                          // the reply as written (the next request sends it back) and its calls
                    const agent::Parsed pc = parse_calls(g.full, tools);
                    done["raw"] = g.full;
                    done["text"] = pc.content;
                    if (!pc.calls.empty()) done["tool_calls"] = pc.calls;
                }
                if (g.checks > 0) done["spec"] = {{"checks", g.checks}, {"drafted", g.drafted}, {"accepted", g.accepted},
                                                  {"looked", g.looked}, {"look_ok", g.look_ok}};
                send(done);
            } catch (const std::exception& e) {
                send({{"type", "error"}, {"message", e.what()}});
            }
            sink.done();
            return true;
        });
    });

    // ---------------- Speed measurement (auto-tune and benchmark) ----------------
    // Text for measuring prompt reading, and for learning expert usage when nothing has been learned yet.
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
    // Generation speed (tokens/second) the way a chat really runs: the same generate() a chat uses (chat format,
    // sampling with these settings, the experts recorded for the dashboard and for learning, speculative decoding
    // with `draft` guesses per check or none, and prompt lookup as set), over three different questions, n tokens each. A fixed seed makes every
    // setting answer the same way, so the settings are compared fairly. Only the reply is timed, not reading the
    // question. This is what the chat shows, minus the browser drawing it.
    Sampling meas_sp;                                                   // set by /api/autotune and /api/bench
    auto measure = [&](int n, int draft) -> double {
        static const char* questions[] = {
            "Explain step by step how a refrigerator keeps food cold, for someone with no science background.",
            "Write a Python function that finds the longest palindrome in a string, with comments, then explain how it works.",
            // A short worked answer: reasoning that repeats the question's numbers is where prompt lookup guesses
            // most often, so its cost (or gain) shows here as it does in a chat.
            "What is 17 x 23? Explain briefly.",
        };
        std::unique_ptr<Speculator> saved = std::move(eng.spec);
        const int saved_seed = eng.seed;
        if (draft > 0) eng.spec = std::make_unique<Speculator>(*eng.model, draft, eng.draft_p_min);
        eng.spec_changed();
        eng.seed = 4242;
        eng.learn = false;
        double toks = 0, secs = 0;
        try {
            {   // a short untimed answer first: right after a load the experts are still coming off the drive into RAM
                eng.model->reset();
                eng.cached.clear();
                generate(json::array({json{{"role", "user"}, {"content", questions[0]}}}), meas_sp, std::max(4, n / 2), {}, nullptr, nullptr);
            }
            for (const char* q : questions) {
                eng.model->reset();
                eng.cached.clear();
                const json msgs = json::array({json{{"role", "user"}, {"content", q}}});
                const Gen g = generate(msgs, meas_sp, std::max(4, n), {}, nullptr, nullptr);
                toks += (double)g.n_gen;
                secs += g.gen_s;
            }
        } catch (...) { eng.spec = std::move(saved); eng.spec_changed(); eng.seed = saved_seed; eng.learn = true; throw; }
        eng.learn = true;
        eng.spec = std::move(saved);
        eng.spec_changed();
        eng.seed = saved_seed;
        eng.model->reset();
        eng.cached.clear();
        return toks / std::max(secs, 1e-9);
    };
    auto sampling_of = [&](const json& b) {
        Sampling sp;
        sp.temperature = b.value("temperature", sp.temperature);
        sp.top_p = b.value("top_p", sp.top_p);
        sp.top_k = b.value("top_k", sp.top_k);
        sp.min_p = b.value("min_p", sp.min_p);
        sp.repeat_penalty = b.value("repeat_penalty", sp.repeat_penalty);
        sp.presence_penalty = b.value("presence_penalty", sp.presence_penalty);
        sp.frequency_penalty = b.value("frequency_penalty", sp.frequency_penalty);
        return sp;
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
    // Each measurement answers three questions the way a chat does, with the chat's sampling settings and a fixed seed
    // (see measure), and times only the replies. Searched one setting at a time, in the order that matters most:
    // split mode (a reload each), then threads, then guesses per check.
    svr.Post("/api/autotune", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("model")) {
            res.status = 400;
            res.set_content("{\"error\":\"expected {\\\"model\\\": \\\"...\\\"}\"}", "application/json; charset=utf-8");
            return;
        }
        const auto file = model_file(body.value("model", std::string()));
        if (file.empty()) {
            res.status = 400;
            res.set_content(dump({{"error", "model not found in the models folder"}}), "application/json; charset=utf-8");
            return;
        }
        if (!mu.try_lock_for(std::chrono::seconds(3))) {
            res.status = 409;
            res.set_content("{\"error\":\"busy: wait for the current reply or load to finish\"}", "application/json; charset=utf-8");
            return;
        }
        auto lock = std::make_shared<std::unique_lock<std::shared_timed_mutex>>(mu, std::adopt_lock);
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
                // Connections are measured last: everything before is measured for one reply at a time.
                const int want_conn = base.parallel;
                const double floor_tps = std::max(0.5, body.value("min_tps", 5.0));
                if (want_conn > 1) { if (base.max_ctx > 0) base.max_ctx /= want_conn; base.parallel = 1; }
                meas_sp = sampling_of(body);                 // measured with the chat's own sampling settings
                base.mtp = has_mtp;                          // the draft layer stays loaded so guesses can be tried

                json results = json::array();
                auto record = [&](const std::string& label, json setting, double tps) {
                    setting["label"] = label;
                    setting["tps"] = std::round(tps * 10) / 10;
                    results.push_back(setting);
                    std::fprintf(stderr, "autotune: %s: %.1f tok/s\n", label.c_str(), tps);
                    return send({{"type", "result"}, {"label", label}, {"tps", setting["tps"]}});
                };
                int n_gen = 48;                               // tokens per question, set from the first measurement
                // per question: about 5 s each, so a setting that pays a cost early (VRAM experts that follow the
                // conversation move in during the first words) is measured over a reply long enough to show what it earns
                auto size_runs = [&](double tps) { n_gen = std::max(64, std::min(480, (int)std::lround(tps * 5))); };

                // ---- 1. split mode (when the model doesn't fit in VRAM) ----
                std::string mode = "auto";
                std::string loaded_mode;
                // Split modes: "layers", "experts" (whole expert layers in VRAM), "hot" (experts in RAM, with the
                // busiest experts in VRAM, from what tierlm has learned about this model), "live" (as hot, then the
                // VRAM experts follow the conversation).
                // A mode ending in "@1" uses only the first GPU (several GPUs: hot experts work with one GPU, and
                // a much slower second card can cost more than its VRAM brings, so both are measured).
                auto load_mode = [&](const std::string& mm) {
                    LoadOptions o = base;
                    const bool one = mm.size() > 2 && mm.compare(mm.size() - 2, 2, "@1") == 0;
                    const std::string m = one ? mm.substr(0, mm.size() - 2) : mm;
                    const bool hot = m == "hot" || m == "live";
                    o.split_mode = hot ? "experts" : m;
                    o.live_hot = m == "live";
                    if (one) o.multi_gpu = false;
                    eng.draft_max = 0;
                    load(file.u8string(), o, t0, seed, hot);
                    loaded_mode = mm;
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
                // Each result is listed with every setting it was measured with (GPU and CPU both work in a split;
                // "6 CPU threads" alone read as if the GPU weren't used).
                const char* mode_label[] = {"GPU + RAM, split by whole layers", "GPU + RAM, experts in RAM",
                                            "GPU + RAM, experts in RAM, busiest experts in VRAM",
                                            "GPU + RAM, experts in RAM, VRAM experts following the conversation"};
                std::string mode_text;                        // the split chosen so far, for the later rows
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
                std::map<std::string, std::pair<double, std::string>> plain_of;   // split mode -> speed without guessing, label
                if (split) {
                    // (hot experts: MoE models, and not yet for experts with biases, gpt-oss)
                    const auto& hp_ = eng.model->hparams();
                    const bool can_hot = hp_.is_moe() && !hp_.expert_bias;
                    const size_t n_gpus = eng.model->placement().gpus.size();
                    std::string gpu0 = n_gpus ? eng.model->placement().gpus[0].name : std::string();   // "CUDA0"...
                    if (ggml_backend_dev_t d = n_gpus ? ggml_backend_dev_by_name(gpu0.c_str()) : nullptr)
                        gpu0 = ggml_backend_dev_description(d);                                        // ...as people know it
                    struct M { std::string mode; int label; };
                    std::vector<M> modes = {{"layers", 0}, {"experts", 1}};
                    if (can_hot && n_gpus <= 1) { modes.push_back({"hot", 2}); modes.push_back({"live", 3}); }   // (hot experts: one GPU)
                    if (n_gpus > 1) {                                // the same on the first GPU alone
                        modes.push_back({"experts@1", 1});
                        if (can_hot) { modes.push_back({"hot@1", 2}); modes.push_back({"live@1", 3}); }
                    }
                    bool no_hot = false;                             // hot experts had no VRAM left: skip live
                    for (const M& md : modes) {
                        const std::string& m = md.mode;
                        const bool one = m.size() > 2 && m.compare(m.size() - 2, 2, "@1") == 0;
                        const std::string base_m = one ? m.substr(0, m.size() - 2) : m;
                        const std::string label = std::string(mode_label[md.label]) + (one ? ", only the " + gpu0 : "");
                        if (base_m == "experts" && one) no_hot = false;
                        if (base_m == "live" && no_hot) continue;
                        if (base_m == "hot" && learned() <= 0) {
                            if (!status("Learning which experts this model uses")) throw std::runtime_error("cancelled");
                            calibrate();
                        }
                        if (loaded_mode != m) {
                            if (!status("Loading with " + label)) throw std::runtime_error("cancelled");
                            load_mode(m);
                        }
                        if (base_m == "live" && !eng.model->live_hot()) continue;   // not enough RAM to spare: loaded as hot
                        if ((base_m == "hot" || base_m == "live") && eng.model->placement().hot_experts == 0) {
                            no_hot = true;                           // nothing to gain: no VRAM left (or several GPUs)
                            continue;
                        }
                        if (!status("Measuring " + label)) throw std::runtime_error("cancelled");
                        // (the busiest experts and the ones that follow the conversation: at least 160 tokens per
                        // question even when whole layers set a short run, since moving experts in pays off over a reply)
                        const double t = measure(base_m == "hot" || base_m == "live" ? std::max(n_gen, 160) : n_gen, 0);
                        if (!record(label + ", " + threads_text(eng.threads) + ", no speculative decoding", {{"split_mode", m}}, t))
                            throw std::runtime_error("cancelled");
                        plain_of[m] = {t, label};
                        if (t > best) { best = t; mode = m; mode_text = label; }
                    }
                    if (loaded_mode != mode) {
                        if (!status("Loading the faster split again")) throw std::runtime_error("cancelled");
                        load_mode(mode);
                    }
                    // Threads and guesses are measured with replies sized from the winner's speed; the winner is
                    // measured again at that length first, so the comparisons after it are like for like (a short
                    // reply and a long one don't run at the same speed).
                    const int n_before = (mode == "hot" || mode == "live") ? std::max(n_gen, 160) : n_gen;
                    size_runs(best);
                    if (n_gen != n_before) {
                        if (!status("Measuring " + mode_text + " again with longer replies")) throw std::runtime_error("cancelled");
                        best = measure(n_gen, 0);
                        if (!record(mode_text + ", " + threads_text(eng.threads) + ", no speculative decoding (again, " +
                                    std::to_string(n_gen) + "-token replies)", {{"split_mode", mode}}, best))
                            throw std::runtime_error("cancelled");
                    }
                } else {
                    if (!status("Measuring")) throw std::runtime_error("cancelled");
                    best = measure(n_gen, 0);
                    mode_text = eng.model->on_gpu() ? "All on the GPU" : "CPU only";
                    record(mode_text + (cpu_work ? ", " + threads_text(t0) : "") + ", no speculative decoding", json::object(), best);
                    mode = "auto";
                }

                // ---- 2. CPU threads (when the CPU does part of the work) ----
                int threads = t0;
                if (cpu_work) {
                    // (half too: with most of a big model's experts in RAM, fewer threads can be faster, they fight less over RAM:
                    // Flash-Next on a 9950X3D ran best at 8-18 of its 32)
                    std::vector<int> cands = {std::max(1, t0 / 2), t0, std::max(1, t0 * 3 / 4), std::min(hw, (t0 * 3 + 1) / 2)};
                    std::sort(cands.begin(), cands.end());
                    cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
                    for (int c : cands) {
                        if (c == t0) continue;               // measured above
                        eng.threads = c;
                        if (!status("Measuring " + threads_text(c))) throw std::runtime_error("cancelled");
                        const double t = measure(n_gen, 0);
                        if (!record(mode_text + ", " + threads_text(c) + ", no speculative decoding", {{"threads", c}}, t)) throw std::runtime_error("cancelled");
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
                        if (!record(mode_text + (cpu_work ? ", " + threads_text(eng.threads) : "") + ", speculative decoding " +
                                    std::to_string(d) + (d == 1 ? " guess" : " guesses"), {{"draft_max", d}}, t))
                            throw std::runtime_error("cancelled");
                        if (t > best) { best = t; draft = d; }
                    }
                    if (draft > 0 && best < plain * 1.03) draft = 0;    // not worth it for a sliver
                    if (draft == 0) best = plain;
                    // Guessing changes which split wins: a check of several guessed words costs about one read per
                    // distinct expert, so with most experts already in VRAM (busiest, or following the conversation)
                    // checks are cheap, and with experts in RAM they aren't. (Flash-Next IQ3_XXS, RTX 5090: experts in
                    // RAM won without guessing, 50.4 against 49.1 tok/s, but lost with it, 47 against 79-87.) So the
                    // VRAM-expert splits within 15% of the winner without guessing are measured with guesses too.
                    if (plain_of.count(mode)) {
                        const auto base_of = [](const std::string& m) {
                            return m.size() > 2 && m.compare(m.size() - 2, 2, "@1") == 0 ? m.substr(0, m.size() - 2) : m;
                        };
                        std::vector<std::string> alts;
                        for (const auto& [m, v] : plain_of) {
                            const std::string b = base_of(m);
                            if (m != mode && (b == "hot" || b == "live") && v.first >= plain_of[mode].first * 0.85) alts.push_back(m);
                        }
                        std::sort(alts.begin(), alts.end(), [&](const std::string& a, const std::string& b) { return plain_of[a].first > plain_of[b].first; });
                        if (alts.size() > 2) alts.resize(2);
                        const int d0 = draft;
                        for (const std::string& alt : alts) {
                            const std::string alt_label = plain_of[alt].second;
                            if (!status("Loading with " + alt_label)) throw std::runtime_error("cancelled");
                            load_mode(alt);
                            eng.threads = threads;                  // (a load starts from the default threads)
                            // the winner's best guesses and one more (checks are cheaper with experts in VRAM); when
                            // guessing didn't pay on the winner, 3 and 4, the usual best with experts in VRAM
                            std::vector<int> tries = d0 > 0 ? std::vector<int>{d0} : std::vector<int>{3, 4};
                            if (d0 > 0 && d0 < 4) tries.push_back(d0 + 1);
                            for (int d : tries) {
                                if (!status("Measuring " + alt_label + ", speculative decoding " + std::to_string(d) + " guesses"))
                                    throw std::runtime_error("cancelled");
                                const double t = measure(n_gen, d);
                                if (!record(alt_label + (cpu_work ? ", " + threads_text(eng.threads) : "") + ", speculative decoding " +
                                            std::to_string(d) + (d == 1 ? " guess" : " guesses"), {{"split_mode", alt}, {"draft_max", d}}, t))
                                    throw std::runtime_error("cancelled");
                                if (t > (draft > 0 ? best : best * 1.03)) { best = t; mode = alt; mode_text = alt_label; draft = d; }
                            }
                        }
                        if (loaded_mode != mode) {
                            if (!status("Loading the faster split again")) throw std::runtime_error("cancelled");
                            load_mode(mode);
                            eng.threads = threads;
                        }
                    }
                }

                // ---- 4. connections (asked for more than one): how fast each reply is with 1, 2, 4... at once ----
                // The most at once that keeps every reply at or above the floor (5 tok/s unless set) is recommended
                // and saved; the setup screen shows the numbers and the person can choose otherwise.
                int conns = 1;
                if (want_conn > 1) {
                    if (!status("Loading for " + std::to_string(want_conn) + " connections")) throw std::runtime_error("cancelled");
                    base.parallel = want_conn;
                    if (base.max_ctx > 0) base.max_ctx *= want_conn;
                    bool loaded_par = true;
                    try { load_mode(mode); }
                    catch (const std::exception& e) {          // (usually: not enough memory for that many contexts)
                        record("Connections: " + std::to_string(want_conn) + " didn't fit (" + e.what() + ")", {{"connections", want_conn}}, 0);
                        base.parallel = 1; if (base.max_ctx > 0) base.max_ctx /= want_conn;
                        load_mode(mode);
                        loaded_par = false;
                    }
                    if (loaded_par) {
                        static const char* qs[] = {
                            "Explain step by step how a refrigerator keeps food cold, for someone with no science background.",
                            "Write a Python function that finds the longest palindrome in a string, with comments, then explain how it works.",
                            "Describe the water cycle and why it matters for farming." };
                        std::vector<int> ks;
                        for (int k = 1; k < want_conn; k *= 2) ks.push_back(k);
                        ks.push_back(want_conn);
                        const int saved_seed = eng.seed;
                        eng.seed = 4242;
                        // More at once must also add speed in all (10% over the best with fewer), not just stay above
                        // the floor: on a fast GPU, small batched steps can cost more than they share, and then one
                        // at a time (the others waiting in line) serves everyone sooner.
                        double best_total = 0;
                        bool floor_ok_more = false;
                        for (int k : ks) {
                            if (!status("Measuring " + std::to_string(k) + (k == 1 ? " reply" : " replies at once"))) throw std::runtime_error("cancelled");
                            // Measured twice, the better run kept: a run is only a second or two on a fast GPU, so
                            // one hiccup (Windows, the browser, a late start) could make it read 25% low.
                            double each = 0, total = 0;
                            for (int rep = 0; rep < 2; ++rep) {
                                for (int sq = 0; sq < (int)eng.slots.size(); ++sq) { eng.model->seq_reset(sq); eng.slots[sq].cached.clear(); }
                                std::vector<Gen> gs(k);
                                std::vector<std::thread> th;
                                std::string err;
                                std::mutex em;
                                for (int i = 0; i < k; ++i)
                                    th.emplace_back([&, i] {
                                        try { gs[i] = generate(json::array({json{{"role", "user"}, {"content", qs[i % 3]}}}), meas_sp, std::max(32, n_gen / 2), {}, nullptr, nullptr); }
                                        catch (const std::exception& e) { std::lock_guard<std::mutex> lk(em); err = e.what(); }
                                    });
                                for (auto& t : th) t.join();
                                if (!err.empty()) throw std::runtime_error(err);
                                double e1 = 1e9, t1 = 0;
                                for (const Gen& g : gs) { const double t = g.n_gen / std::max(g.gen_s, 1e-9); e1 = std::min(e1, t); t1 += t; }
                                if (t1 > total) { total = t1; each = e1; }
                            }
                            if (!record("Connections: " + std::to_string(k) + " at once, " + std::to_string((int)std::lround(total)) + " tok/s in all, slowest",
                                        {{"connections", k}, {"total_tps", std::round(total * 10) / 10}}, each)) throw std::runtime_error("cancelled");
                            if (each >= floor_tps && (k == 1 || total >= best_total * 1.10)) { conns = k; best_total = total; }
                            else if (each >= floor_tps) floor_ok_more = true;
                            if (k == 1) best_total = total;
                        }
                        if (floor_ok_more && conns < want_conn)
                            record("Recommended: " + std::to_string(conns) + (conns == 1 ? " connection" : " connections") +
                                   " (more at once didn't add speed on this PC), in all", json::object(), best_total);
                        eng.seed = saved_seed;
                        for (int sq = 0; sq < (int)eng.slots.size(); ++sq) { eng.model->seq_reset(sq); eng.slots[sq].cached.clear(); }
                        if (conns != want_conn) {
                            if (!status("Loading with " + std::to_string(conns) + (conns == 1 ? " connection" : " connections"))) throw std::runtime_error("cancelled");
                            if (base.max_ctx > 0) base.max_ctx = base.max_ctx / want_conn * conns;
                            base.parallel = conns;
                            load_mode(mode);
                        }
                    }
                }

                // ---- the winner: keep it loaded and remember it ----
                eng.draft_max = draft > 0 ? draft : 0;
                eng.spec.reset();
                if (draft > 0) eng.spec = std::make_unique<Speculator>(*eng.model, draft, eng.draft_p_min);
                else eng.draft_max = eng.model->hparams().is_moe() ? 2 : 3;
                eng.spec_changed();
                eng.model->reset();
                eng.cached.clear();
                const std::string id = body.value("model", std::string());
                json saved = body;
                saved.erase("model");
                const bool one_gpu = mode.size() > 2 && mode.compare(mode.size() - 2, 2, "@1") == 0;
                const std::string mode0 = one_gpu ? mode.substr(0, mode.size() - 2) : mode;
                saved["split_mode"] = split ? (mode0 == "hot" || mode0 == "live" ? "experts" : mode0) : body.value("split_mode", std::string("auto"));
                saved["hot_experts"] = mode0 == "hot" || mode0 == "live";      // on only where it measured fastest
                saved["live_experts"] = mode0 == "live";
                saved["multi_gpu"] = !one_gpu;                                 // several GPUs: whether using them all won
                saved["threads"] = threads;
                saved["speculative"] = draft > 0;
                if (draft > 0) saved["draft_max"] = draft;
                saved["settings_v"] = 2;
                if (want_conn > 1) { saved["connections"] = conns; saved["min_tps"] = floor_tps; }
                // when, with which tierlm and on what hardware: shown with the results next time this model is picked
                std::string pc;
                for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
                    ggml_backend_dev_t dv = ggml_backend_dev_get(d);
                    if (ggml_backend_dev_type(dv) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
                    const std::string nm = ggml_backend_dev_description(dv);
                    if (pc.find(nm) != std::string::npos) continue;      // (the same card through CUDA and Vulkan)
                    if (!pc.empty()) pc += " + ";
                    pc += nm;
                }
                {
                    size_t fr = 0, tot = 0;
                    if (ggml_backend_dev_t cd = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) ggml_backend_dev_memory(cd, &fr, &tot);
                    if (!pc.empty()) pc += ", ";
                    pc += std::to_string((tot + (512ull << 20)) >> 30) + " GB RAM";
                }
                const std::time_t now = std::time(nullptr);
                char when[32] = {0};
                std::strftime(when, sizeof(when), "%Y-%m-%d %H:%M", std::localtime(&now));
                saved["tuned"] = {{"tps", std::round(best * 10) / 10}, {"results", results}, {"when", when},
                                  {"version", TIERLM_VERSION}, {"pc", pc}};
                settings["models"][id] = saved;
                settings["last_model"] = id;
                write_settings(a.settings, settings);
                send({{"type", "done"}, {"tps", std::round(best * 10) / 10}, {"settings", saved}, {"results", results},
                      {"all_on_gpu", eng.model && eng.model->placement().all_on_gpu()}});
            } catch (const std::exception& e) {
                send({{"type", "error"}, {"message", e.what()}});
            }
            sink.done();
            return true;
        });
    });


    // ---------------- Benchmark: the loaded model's speed with its current settings ----------------
    svr.Post("/api/bench", [&](const httplib::Request& req, httplib::Response& res) {
        if (!mu.try_lock_for(std::chrono::seconds(3))) {
            res.status = 409;
            res.set_content("{\"error\":\"busy: wait for the current reply or load to finish\"}", "application/json; charset=utf-8");
            return;
        }
        std::lock_guard<std::shared_timed_mutex> lk(mu, std::adopt_lock);
        if (!eng.model) { res.status = 409; res.set_content("{\"error\":\"no model loaded\"}", "application/json; charset=utf-8"); return; }
        try {
            std::fprintf(stderr, "benchmark: reading 512 tokens, then answering three questions like a chat\n");
            const json bb = json::parse(req.body.empty() ? std::string("{}") : req.body, nullptr, false);
            meas_sp = sampling_of(bb.is_object() ? bb : json::object());
            const double p = measure_prompt(512);
            const int draft = eng.spec ? eng.draft_max : 0;
            measure(4, draft);                                  // warm-up
            const double g = measure(160, draft);
            eng.model->reset();
            eng.cached.clear();
            std::fprintf(stderr, "benchmark: prompt %.1f tok/s, generation %.1f tok/s\n", p, g);
            res.set_content(dump({{"prompt_tps", std::round(p * 10) / 10}, {"gen_tps", std::round(g * 10) / 10},
                                  {"speculative", draft > 0}, {"draft_max", draft}, {"threads", eng.threads}}), "application/json; charset=utf-8");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(dump({{"error", e.what()}}), "application/json; charset=utf-8");
        }
    });

    // ---------------- Speculative decoding on/off without reloading (when the draft layer is loaded) ----------------
    svr.Post("/api/spec", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) { res.status = 400; res.set_content("{\"error\":\"bad request\"}", "application/json; charset=utf-8"); return; }
        // A quick change: waits in line behind a running reply instead of answering "busy".
        if (!line.enter(mu, req.is_connection_closed)) { res.status = 503; res.set_content("{\"error\":\"busy\"}", "application/json; charset=utf-8"); return; }
        std::lock_guard<std::shared_timed_mutex> lk(mu, std::adopt_lock);
        if (!eng.model) { res.status = 409; res.set_content("{\"error\":\"no model loaded\"}", "application/json; charset=utf-8"); return; }
        // Prompt lookup alone changes nothing in the model's cache: no need to start over.
        if (body.contains("lookup") && body["lookup"].is_number_integer()) eng.lookup = std::max(-1, std::min(16, body["lookup"].get<int>()));
        if (!body.contains("speculative") && !body.contains("draft_max")) {
            eng.spec_changed();                                 // the next reply brings lookup up to date with the cache
            std::fprintf(stderr, "prompt lookup: %s\n", eng.lookup_len() > 0 ? ("up to " + std::to_string(eng.lookup_len()) + " tokens per check").c_str() : "off");
            res.set_content(dump({{"ok", true}, {"speculative", eng.spec != nullptr}, {"draft_max", eng.draft_max}, {"lookup", eng.lookup_len()}}), "application/json; charset=utf-8");
            return;
        }
        if (body.contains("draft_p_min") && body["draft_p_min"].is_number()) eng.draft_p_min = std::max(0.0f, std::min(1.0f, body["draft_p_min"].get<float>()));
        const bool on = body.value("speculative", eng.spec != nullptr);
        const int d = std::min(8, std::max(1, body.value("draft_max", eng.draft_max > 0 ? eng.draft_max : 2)));
        if (on && !eng.model->has_mtp()) {
            res.status = 409;
            res.set_content(dump({{"error", "this model was loaded without its draft layer: turn on Speculative decoding on the setup screen and load it again"}}), "application/json; charset=utf-8");
            return;
        }
        eng.draft_max = d;
        eng.spec.reset();
        if (on) eng.spec = std::make_unique<Speculator>(*eng.model, d, eng.draft_p_min);
        eng.spec_changed();
        eng.cached.clear();                                     // the draft block's cache starts over with the next reply
        eng.model->reset();
        std::fprintf(stderr, "speculative decoding %s%s\n", on ? "on, " : "off", on ? (std::to_string(d) + (d == 1 ? " guess per check" : " guesses per check")).c_str() : "");
        if (eng.lookup_len() > 0) std::fprintf(stderr, "prompt lookup: up to %d tokens per check\n", eng.lookup_len());
        res.set_content(dump({{"ok", true}, {"speculative", on}, {"draft_max", d}, {"lookup", eng.lookup_len()}}), "application/json; charset=utf-8");
    });

    // ---------------- Forget what was learned about the loaded model's experts ----------------
    svr.Post("/api/usage/reset", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::shared_timed_mutex> lk(mu);
        std::fill(eng.usage.begin(), eng.usage.end(), 0.0f);
        std::error_code ec;
        if (!eng.usage_path.empty()) fs::remove(fs::u8path(eng.usage_path), ec);
        eng.usage_dirty = false;
        std::fprintf(stderr, "expert usage for this model forgotten; the next load places whole expert layers again\n");
        res.set_content("{\"ok\":true}", "application/json; charset=utf-8");
    });

    // ---------------- Loading progress and the console (no lock: they answer while a model loads) ----------------
    svr.Get("/api/progress", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(dump({{"loading", loading.load()}, {"progress", load_progress.load()}}), "application/json; charset=utf-8");
    });
    svr.Get("/api/log", [&](const httplib::Request& req, httplib::Response& res) {
        long long after = -1;
        if (req.has_param("after")) after = std::atoll(req.get_param_value("after").c_str());
        json lines = json::array();
        long long next = 0;
        {
            std::lock_guard<std::mutex> lk(g_log.mu);
            next = g_log.next;
            for (const auto& l : g_log.lines)
                if (l.n > after) lines.push_back({{"n", l.n}, {"time", l.time}, {"text", l.text}, {"v", verbose_line(l.text)}});
        }
        res.set_content(dump({{"lines", lines}, {"next", next}}), "application/json; charset=utf-8");
    });

    // ---------------- The embedding model (a second model, next to the chat model) ----------------
    // Embedding models are small and run on their own: requests for embeddings don't wait for a chat reply to
    // finish. The two models take turns on the GPU between steps (device_mutex), so neither stalls for long.
    // (the slot itself, `emb`, is declared before the model routes: Eject frees it too)
    // For looking back at a chat's archive by meaning (chats::recall): one text to a unit vector with the embedding
    // model, if one is loaded. Nomic's models are trained with "search_query: " / "search_document: " in front.
    embed_for_recall = [&](const std::string& text, bool query) -> std::vector<float> {
        std::lock_guard<std::mutex> lk(emb.mu);
        if (!emb.enc) return {};
        const bool nomic = emb.enc->arch() == "nomic-bert";
        std::vector<int32_t> ids, body_ids = emb.tok.encode((nomic ? (query ? "search_query: " : "search_document: ") : "") + text);
        const int room = emb.enc->max_tokens() - (emb.tok.cls() >= 0) - (emb.tok.sep() >= 0);
        if ((int)body_ids.size() > room) body_ids.resize(std::max(0, room));
        if (emb.tok.cls() >= 0) ids.push_back(emb.tok.cls());
        ids.insert(ids.end(), body_ids.begin(), body_ids.end());
        if (emb.tok.sep() >= 0) ids.push_back(emb.tok.sep());
        std::vector<float> v = emb.enc->embed(ids, eng.threads);
        double norm = 0;
        for (float x : v) norm += (double)x * x;
        norm = std::sqrt(norm);
        if (norm > 0) for (float& x : v) x = (float)(x / norm);
        return v;
    };
    auto embed_info = [&]() -> json {        // (emb.mu held)
        if (!emb.enc) return {{"loaded", false}};
        return {{"loaded", true}, {"id", emb.id}, {"name", emb.name}, {"arch", emb.enc->arch()}, {"dims", emb.enc->n_embd()},
                {"max_tokens", emb.enc->max_tokens()}, {"device", emb.enc->device_name()}, {"on_gpu", emb.enc->on_gpu()},
                {"bytes", emb.enc->weight_bytes()}, {"load_s", emb.load_s},
                {"pooling", emb.enc->pooling() == Encoder::Pooling::Cls ? "cls" : emb.enc->pooling() == Encoder::Pooling::Last ? "last" : "mean"}};
    };
    auto embed_load = [&](const fs::path& file, const std::string& id) {      // (emb.mu held)
        const auto t0 = Clock::now();
        emb.enc.reset();
        // next to the chat model on its own GPU when there's room, else the CPU (never another, slower chip)
        std::string where = "auto";
        if (eng.model && eng.model->on_gpu() && !eng.model->placement().gpus.empty()) where = "auto:" + eng.model->placement().gpus[0].name;
        auto enc = Encoder::load(file.u8string(), where);
        emb.tok = Tokenizer::from_gguf(file.u8string());
        if (!emb.tok.wordpiece()) throw std::runtime_error("this embedding model's tokenizer is not supported yet");
        emb.enc = std::move(enc);
        emb.id = id; emb.path = file.u8string();
        const json d = describe_model(emb.path);
        emb.name = d.value("name", std::string()).empty() ? file.stem().u8string() : d.value("name", std::string());
        emb.load_s = since(t0);
        std::fprintf(stderr, "embedding model: %s (%s, %d numbers per text, up to %d tokens) on %s, loaded in %.1f s\n", emb.name.c_str(),
                     emb.enc->arch().c_str(), emb.enc->n_embd(), emb.enc->max_tokens(), emb.enc->device_name().c_str(), emb.load_s);
    };

    // ---------------- OpenAI-compatible API ----------------
    const long long started = (long long)std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    // The loaded model's id: its path inside the models folder, as /v1/models lists it.
    auto loaded_id = [&]() -> std::string {
        if (eng.path.empty()) return {};
        std::error_code ec;
        return model_key(eng.path);
    };

    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json data = json::array();
        std::string cur;
        {
            std::unique_lock<std::shared_timed_mutex> lk(mu, std::try_to_lock);
            cur = lk.owns_lock() ? loaded_id() : last_info.value("path", std::string());
            if (!lk.owns_lock() && !cur.empty()) cur = fs::u8path(cur).filename().u8string();
        }
        if (!cur.empty()) {
            json m = {{"id", cur}, {"object", "model"}, {"created", started}, {"owned_by", "tierlm"}, {"loaded", true}, {"type", "llm"}};
            std::shared_lock<std::shared_timed_mutex> lk(mu, std::try_to_lock);
            if (lk.owns_lock() && eng.vision) { m["type"] = "vlm"; m["capabilities"] = json::array({"vision"}); }   // (as LM Studio says it)
            data.push_back(m);
        }
        {
            std::lock_guard<std::mutex> lk(emb.mu);
            if (emb.enc) data.push_back({{"id", emb.id}, {"object", "model"}, {"created", started}, {"owned_by", "tierlm"}, {"loaded", true}, {"type", "embeddings"}});
        }
        bool exists = false;
        for (const auto& m : scan_models(exists)) {
            if (!m.value("supported", false) || m.value("id", std::string()) == cur) continue;
            data.push_back({{"id", m["id"]}, {"object", "model"}, {"created", started}, {"owned_by", "tierlm"}, {"loaded", false}});
        }
        res.set_content(dump({{"object", "list"}, {"data", data}}), "application/json; charset=utf-8");
    });

    // LM Studio's own REST API (/api/v0): apps written for LM Studio list the models here, with what each one is
    // ("llm", "vlm" for one that reads images, "embeddings") and whether it is loaded, and some send their requests
    // to /api/v0/chat/completions and /api/v0/embeddings (the same as /v1). The fields are LM Studio's.
    auto v0_model = [&](const std::string& id, const std::string& type, bool loaded, const std::string& arch, int64_t ctx_max, int64_t ctx_loaded) {
        const auto slash = id.find('/');
        std::string quant;                                    // from the file name, as LM Studio shows it: Q4_K_M, Q8_0, MXFP4...
        static const std::regex qre(R"((?:^|[-_.])((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|MXFP4|BF16|F16|F32|\d(?:\.\d)?bpw))(?:[-_.]|$))", std::regex::icase);
        std::smatch mq;
        const std::string stem = fs::u8path(id).stem().u8string();
        if (std::regex_search(stem, mq, qre)) quant = mq[1].str();
        json m = {{"id", id}, {"object", "model"}, {"type", type}, {"publisher", slash == std::string::npos ? std::string("local") : id.substr(0, slash)},
                  {"arch", arch}, {"compatibility_type", "gguf"}, {"quantization", quant}, {"state", loaded ? "loaded" : "not-loaded"},
                  {"max_context_length", ctx_max}};
        if (loaded && ctx_loaded > 0) m["loaded_context_length"] = ctx_loaded;
        if (type == "vlm") m["capabilities"] = json::array({"vision"});
        return m;
    };
    auto v0_models = [&]() {
        json data = json::array();
        std::string cur;
        json cur_m;
        {
            std::shared_lock<std::shared_timed_mutex> lk(mu, std::try_to_lock);
            if (lk.owns_lock() && eng.model) {
                cur = loaded_id();
                cur_m = v0_model(cur, eng.vision ? "vlm" : "llm", true, eng.model->arch(), 0,
                                 eng.slots.empty() ? eng.model->max_ctx() : eng.ctx_per);
            } else if (!lk.owns_lock()) {                     // (busy writing a reply: what was loaded last)
                cur = last_info.value("path", std::string());
                if (!cur.empty()) {
                    cur = fs::u8path(cur).filename().u8string();
                    cur_m = v0_model(cur, last_info.value("vision", false) ? "vlm" : "llm", true, last_info.value("arch", std::string()),
                                     0, last_info.value("ctx", 0));
                }
            }
        }
        if (!cur_m.is_null()) data.push_back(cur_m);
        std::string emb_id;
        {
            std::lock_guard<std::mutex> lk(emb.mu);
            if (emb.enc) { emb_id = emb.id; data.push_back(v0_model(emb.id, "embeddings", true, emb.enc->arch(), 0, 0)); }
        }
        bool exists = false;
        for (const auto& m : scan_models(exists)) {
            const std::string id = m.value("id", std::string());
            for (auto& d : data)                              // the loaded ones: the context the model was trained for
                if (d["id"] == id) d["max_context_length"] = m.value("ctx_train", 0);
            if (id == cur || id == emb_id) continue;
            const bool embedding = m.value("kind", std::string()) == "embedding" && m.value("embedding", false);
            if (!embedding && !m.value("supported", false)) continue;
            std::string type = embedding ? "embeddings" : "llm";
            if (!embedding && m.value("arch", std::string()) != "qwen4exp") {   // a vision file next to it: it reads images once loaded
                std::error_code ec;
                const fs::path dir = model_file(id).parent_path();
                for (const auto& f : fs::directory_iterator(dir, ec)) {
                    const std::string n = f.path().filename().u8string();
                    std::string low = n; std::transform(low.begin(), low.end(), low.begin(), ::tolower);
                    if (low.find("mmproj") != std::string::npos && f.path().extension() == ".gguf") { type = "vlm"; break; }
                }
            }
            data.push_back(v0_model(id, type, false, m.value("arch", std::string()), m.value("ctx_train", 0), 0));
        }
        return data;
    };
    svr.Get("/api/v0/models", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(dump({{"object", "list"}, {"data", v0_models()}}), "application/json; charset=utf-8");
    });
    svr.Get(R"(/api/v0/models/(.+))", [&](const httplib::Request& req, httplib::Response& res) {
        const std::string want = req.matches[1].str();
        for (const auto& m : v0_models())
            if (m["id"] == want) return res.set_content(dump(m), "application/json; charset=utf-8");
        oai_error(res, 404, "no model \"" + want + "\" in the models folder", "invalid_request_error", "model_not_found");
    });

    svr.Post(R"(/(?:v1|api/v0)/chat/completions)", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("messages") || !body["messages"].is_array())
            return oai_error(res, 400, "expected a JSON body with a \"messages\" array");
        if (body.value("n", 1) != 1) return oai_error(res, 400, "only n = 1 is supported");
        // Wait in line for replies already asked for (from the web page or other apps): one at a time, or as many
        // at once as there are connections. Loading a different model needs the engine to itself.
        const std::string want = body.value("model", std::string());
        // (another model only if it is one in the models folder: many apps send a fixed name such as "gpt-4o"; and
        // only for apps on this computer: an app elsewhere uses the loaded model, as switching models is a setting)
        const bool other_model = !want.empty() && want != loaded_id() && is_local(req) && [&] {
            std::error_code ec;
            const auto file = model_file(want);
            return !file.empty() && file.extension() == ".gguf";
        }();
        auto lock = std::make_shared<Hold>();
        if (!line.enter_reply(mu, *lock, other_model ? std::function<std::atomic<int>*()>() : std::function<std::atomic<int>*()>(reply_slots),
                              req.is_connection_closed))
            return oai_error(res, 503, "busy: waited too long for earlier replies to finish");
        // The model asked for: the loaded one if the name matches it, or is unknown (many apps send a fixed
        // name such as "gpt-4o"); a different model in the models folder is loaded first, with its saved settings.
        if (other_model && want != loaded_id()) {
            const auto file = model_file(want);
            const bool in_folder = !file.empty() && file.extension() == ".gguf";
            if (in_folder) {
                try {
                    json saved = settings.value("models", json::object()).value(model_key(file.u8string()), json::object());
                    eng.draft_max = std::min(8, std::max(0, saved.value("draft_max", 0)));
                    load(file.u8string(), load_options(saved), saved.value("threads", a.threads), saved.value("seed", -1), saved.value("hot_experts", false));
                    settings["last_model"] = model_key(file.u8string());
                    write_settings(a.settings, settings);
                } catch (const std::exception& e) {
                    eng.spec.reset(); eng.look.reset(); eng.vision.reset(); eng.model.reset(); eng.path.clear();
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
        sp.presence_penalty = body.value("presence_penalty", sp.presence_penalty);
        sp.frequency_penalty = body.value("frequency_penalty", sp.frequency_penalty);
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
        json messages = normalize_messages(body["messages"]);
        // Tools (coding agents such as opencode, Cline or Cursor): the model may answer with tool calls, which the
        // app runs and sends back as "tool" messages. tool_choice "none" leaves them out.
        agent::ojson tools = agent::ojson::array();                 // in the order the app wrote them
        {
            const agent::ojson ob = agent::ojson::parse(req.body, nullptr, false);
            if (ob.is_object() && ob.contains("tools") && ob["tools"].is_array())
                for (const auto& t : ob["tools"])
                    if (agent::tool_function(t).is_object() && agent::tool_function(t).contains("name")) tools.push_back(t);
        }
        if (body.contains("tool_choice") && body["tool_choice"].is_string() && body["tool_choice"].get<std::string>() == "none") tools = agent::ojson::array();
        const bool stream = body.value("stream", false);
        const bool usage_in_stream = body.contains("stream_options") && body["stream_options"].is_object() &&
                                     body["stream_options"].value("include_usage", false);
        const std::string model_id = loaded_id();
        const long long created = (long long)std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        static std::atomic<unsigned> counter{0};
        const std::string id = "chatcmpl-tierlm-" + std::to_string(created) + "-" + std::to_string(++counter);
        auto finish = [](const std::string& stop) { return stop == "length" || stop == "context" ? "length" : "stop"; };
        // The reply's tool calls in OpenAI's form: arguments as a JSON string, each with an id.
        auto openai_calls = [id](const json& calls) {
            json out = json::array();
            for (size_t i = 0; i < calls.size(); ++i)
                out.push_back({{"id", "call_" + id.substr(id.size() > 8 ? id.size() - 8 : 0) + "_" + std::to_string(i)}, {"type", "function"},
                               {"function", {{"name", calls[i]["name"]}, {"arguments", calls[i].value("arguments_json", dump(calls[i]["arguments"]))}}}});
            return out;
        };

        // Too long for the context: with "Drop the oldest messages" on (API panel, "api_trim_oldest"), the oldest
        // messages are left out (down to about 60% of the context); otherwise the app gets an error and can compact
        // the conversation itself, as coding agents do.
        if (settings.value("api_trim_oldest", false) && eng.model) {
            const int ctx = eng.slots.empty() ? eng.model->max_ctx() : eng.ctx_per;
            const int reserve = max_tokens > 0 ? std::min(max_tokens, ctx / 2) : std::max(256, ctx / 8);
            if (prompt_tokens(messages, tools) > ctx - reserve) {
                const int n = drop_oldest(messages, tools, std::min(ctx * 6 / 10, ctx - reserve));
                if (n > 0) std::fprintf(stderr, "api: left out the %d oldest messages to fit the context\n", n);
            }
        }

        if (!stream) {
            try {
                tl_image_tokens = body.value("image_tokens", 0);   // image detail (0: automatic)
                take_think(body);
                Gen g = generate(messages, sp, max_tokens, stops, nullptr, nullptr, tools);
                json msg = {{"role", "assistant"}, {"content", g.text}};
                std::string fin = finish(g.stop);
                // The model's thinking goes in reasoning_content, the answer in content, as LM Studio and OpenAI's
                // reasoning models send them: apps that show `content` show just the answer (even when the reply was
                // cut off while still thinking: then content is empty).
                {
                    const auto [reasoning, answer] = agent::split_thinking(g.text);
                    msg["content"] = answer;
                    if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
                }
                if (!tools.empty()) {
                    const agent::Parsed pc = parse_calls(g.full, tools);
                    const auto [reasoning, answer] = agent::split_thinking(pc.calls.empty() ? g.text : pc.content);
                    msg["content"] = answer;
                    if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
                    if (!pc.calls.empty()) {
                        if (answer.empty()) msg["content"] = nullptr;
                        msg["tool_calls"] = openai_calls(pc.calls);
                        fin = "tool_calls";
                    }
                }
                res.set_content(dump({{"id", id}, {"object", "chat.completion"}, {"created", created}, {"model", model_id},
                    {"choices", json::array({{{"index", 0}, {"message", msg},
                                              {"logprobs", nullptr}, {"finish_reason", fin}}})},
                    {"usage", {{"prompt_tokens", g.n_prompt}, {"completion_tokens", g.n_gen}, {"total_tokens", g.n_prompt + g.n_gen}}}}),
                    "application/json; charset=utf-8");
            } catch (const std::exception& e) {
                oai_error(res, 500, e.what(), "server_error");
            }
            return;
        }
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream", [&, messages, tools, openai_calls, sp, max_tokens, stops, usage_in_stream, model_id, created, id, lock]
                                                              (size_t, httplib::DataSink& sink) {
            auto send = [&](const json& j) { std::string s = "data: " + dump(j) + "\n\n"; return sink.write(s.data(), s.size()); };
            auto chunk = [&](const json& delta, const json& fin) {
                return send({{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", model_id},
                             {"choices", json::array({{{"index", 0}, {"delta", delta}, {"logprobs", nullptr}, {"finish_reason", fin}}})}});
            };
            try {
                chunk({{"role", "assistant"}, {"content", ""}}, nullptr);
                agent::ThinkSplit split;                            // thinking goes out as reasoning_content (as above)
                auto send_parts = [&](const std::pair<std::string, std::string>& p) {
                    bool ok = true;
                    if (!p.first.empty()) ok = chunk({{"reasoning_content", p.first}}, nullptr);
                    if (ok && !p.second.empty()) ok = chunk({{"content", p.second}}, nullptr);
                    return ok;
                };
                tl_image_tokens = body.value("image_tokens", 0);   // image detail (0: automatic)
                take_think(body);
                Gen g = generate(messages, sp, max_tokens, stops, nullptr,
                                 [&](const std::string& delta, double, const std::vector<int32_t>&) {
                                     if (delta.empty()) return true;
                                     return send_parts(split.feed(delta));
                                 }, tools);
                if (g.stop != "abort") send_parts(split.feed("", true));
                if (g.stop != "abort") {
                    std::string fin = finish(g.stop);
                    if (!tools.empty()) {
                        const agent::Parsed pc = parse_calls(g.full, tools);
                        if (!pc.calls.empty()) {
                            json calls = openai_calls(pc.calls);
                            for (size_t i = 0; i < calls.size(); ++i) calls[i]["index"] = (int)i;
                            chunk({{"tool_calls", calls}}, nullptr);
                            fin = "tool_calls";
                        }
                    }
                    chunk(json::object(), fin);
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

    // The page asks /api/info what it may show: from another device it says so ("remote"), and the model's
    // location on the disk is left out (its file name is enough there).
    svr.set_post_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
        if (req.path != "/api/info" || is_local(req) || res.status != 200) return;
        json j = json::parse(res.body, nullptr, false);
        if (!j.is_object()) return;
        j["remote"] = true;
        for (const char* k : {"path", "vision_file", "ssd_file"})
            if (j.contains(k) && j[k].is_string()) j[k] = fs::u8path(j[k].get<std::string>()).filename().u8string();
        res.set_content(dump(j), "application/json; charset=utf-8");
    });

    svr.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
        const bool local = is_local(req);
        if (!local && !allow_network && !a.host_given) {
            oai_error(res, 403, "tierlm only accepts connections from this computer: turn on \"Allow other devices\" in tierlm's settings",
                      "permission_error");
            return httplib::Server::HandlerResponse::Handled;
        }
        // Other devices use the model (chat, the API) but don't run the computer: loading or unloading models,
        // settings, the tuner and the benchmark, the models folder, the embedding model and the console are for
        // the computer running tierlm only (the workspace, saved chats and network settings check this themselves).
        // Refused here, so a script can't do what the page doesn't offer either.
        if (!local && remote_refused(req)) {
            oai_error(res, 403, "settings can only be changed on the computer running tierlm", "permission_error");
            return httplib::Server::HandlerResponse::Handled;
        }
        std::string key;
        { std::lock_guard<std::mutex> lk(net_mu); key = api_key; }
        const bool is_api = req.path.compare(0, 4, "/v1/") == 0 || req.path.compare(0, 5, "/api/") == 0;
        const bool needs_key = !key.empty() && is_api && (!local || req.path.compare(0, 4, "/v1/") == 0 || req.path.compare(0, 8, "/api/v0/") == 0);
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
                  {"api_key_set", !key.empty()}, {"urls", urls}, {"fixed_by_command_line", a.host_given || a.api_key_given},
                  {"api_trim_oldest", settings.value("api_trim_oldest", false)}};
        if (show_key) j["api_key"] = key;
        return j;
    };
    svr.Get("/api/network", [&](const httplib::Request& req, httplib::Response& res) {
        res.set_content(dump(network_info(is_local(req))), "application/json; charset=utf-8");   // the key itself only to this computer
    });
    // Change network access or the API key. Only from this computer: someone on the network mustn't be able to
    // open it wider or change the key.
    svr.Post("/api/network", [&](const httplib::Request& req, httplib::Response& res) {
        if (!is_local(req)) { oai_error(res, 403, "network settings can only be changed on the computer running tierlm", "permission_error"); return; }
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) { oai_error(res, 400, "bad request"); return; }
        const std::string before = bind_host();
        if (body.contains("allow_network")) { allow_network = body.value("allow_network", false); settings["allow_network"] = (bool)allow_network; }
        if (body.contains("api_trim_oldest")) settings["api_trim_oldest"] = body.value("api_trim_oldest", false);
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
        res.set_content(dump(network_info(true)), "application/json; charset=utf-8");
    });

    // ---------------- the workspace (tierlm's own chat page only) ----------------
    // A folder the model can read and, with the user's approval of each change, write. Only for the page on the
    // computer running tierlm: the request must come from this computer (not through a tunnel), be addressed to
    // 127.0.0.1 / localhost (so another website that points its own name at 127.0.0.1 can't use it), come from
    // tierlm's own page if a browser says where from (Origin), and carry a header other websites can't send
    // without asking first (X-Tierlm-Workspace).
    auto ws_check = [&](const httplib::Request& req, bool need_header) -> std::string {
        if (!is_local(req)) return "the workspace can only be used on the computer running tierlm";
        const std::string port = std::to_string(a.port), host = req.get_header_value("Host");
        if (host != "127.0.0.1:" + port && host != "localhost:" + port && host != "[::1]:" + port)
            return "the workspace can only be used from http://127.0.0.1:" + port;
        if (req.has_header("Origin")) {
            const std::string o = req.get_header_value("Origin");
            if (o != "http://127.0.0.1:" + port && o != "http://localhost:" + port && o != "http://[::1]:" + port)
                return "the workspace can only be used from tierlm's own page";
        }
        if (need_header && req.get_header_value("X-Tierlm-Workspace") != "1") return "missing X-Tierlm-Workspace header";
        return "";
    };
    auto ws_info = [&](const httplib::Request& req) {
        const std::string folder = settings.value("workspace_folder", std::string());
        std::error_code ec;
        return json{{"available", ws_check(req, false).empty()}, {"folder", folder},
                    {"exists", !folder.empty() && fs::is_directory(fs::u8path(folder), ec)},
                    {"picker", (bool)
#ifdef _WIN32
                                true
#else
                                false
#endif
                    }};
    };
    svr.Get("/api/workspace", [&](const httplib::Request& req, httplib::Response& res) {
        res.set_content(dump(ws_info(req)), "application/json; charset=utf-8");
    });
    svr.Post("/api/workspace", [&](const httplib::Request& req, httplib::Response& res) {    // {"folder": "C:\\code\\project"}
        if (const std::string why = ws_check(req, true); !why.empty()) return oai_error(res, 403, why, "permission_error");
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) return oai_error(res, 400, "bad request");
        std::string folder = agent::trim(body.value("folder", std::string()));
        std::error_code ec;
        if (!folder.empty()) {
            const fs::path p = fs::weakly_canonical(fs::u8path(folder), ec);
            if (ec || !fs::is_directory(p, ec)) return oai_error(res, 400, "not a folder: " + folder);
            folder = p.u8string();
        }
        settings["workspace_folder"] = folder;
        write_settings(a.settings, settings);
        std::fprintf(stderr, "workspace: %s\n", folder.empty() ? "none" : folder.c_str());
        res.set_content(dump(ws_info(req)), "application/json; charset=utf-8");
    });
    svr.Post("/api/workspace/pick", [&](const httplib::Request& req, httplib::Response& res) {
        if (const std::string why = ws_check(req, true); !why.empty()) return oai_error(res, 403, why, "permission_error");
#ifdef _WIN32
        bool cancelled = true;
        const std::string f = pick_folder_windows(settings.value("workspace_folder", std::string()), cancelled);
        res.set_content(dump({{"cancelled", cancelled}, {"folder", f}}), "application/json; charset=utf-8");
#else
        oai_error(res, 501, "type the folder path instead (the picker is Windows only)");
#endif
    });
    // Runs a tool the model asked for: {"name", "arguments": {...}, "apply": false, "hash": ""}. A change (write_file,
    // edit_file) comes back as a diff unless apply is true. Failures come back as {"error"}: the page passes them to
    // the model, which can try again.
    svr.Post("/api/workspace/tool", [&](const httplib::Request& req, httplib::Response& res) {
        if (const std::string why = ws_check(req, true); !why.empty()) return oai_error(res, 403, why, "permission_error");
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) return oai_error(res, 400, "bad request");
        const std::string folder = settings.value("workspace_folder", std::string());
        try {
            if (folder.empty()) throw workspace::Error("no workspace folder is chosen");
            json args = body.contains("arguments") ? body["arguments"] : json::object();
            if (args.is_string()) args = json::parse(args.get<std::string>(), nullptr, false);
            if (!args.is_object()) throw workspace::Error("the tool's arguments are not a JSON object");
            const std::string name = body.value("name", std::string());
            json r = workspace::run(fs::u8path(folder), name, args, body.value("apply", false), body.value("hash", std::string()));
            if (r.contains("result") && (body.value("apply", false) || !workspace::changes_files(name)))
                std::fprintf(stderr, "workspace: %s %s\n", name.c_str(), workspace::arg_str(args, "path").c_str());
            res.set_content(dump(r), "application/json; charset=utf-8");
        } catch (const std::exception& e) {
            res.set_content(dump({{"error", e.what()}}), "application/json; charset=utf-8");
        }
    });

    // ---------------- Embeddings ----------------
    // GET /api/embed: the loaded embedding model. POST /api/embed/load {"model": id}, POST /api/embed/unload.
    // The choice is saved, and the model loads again when tierlm starts.
    auto save_embed_choice = [&](const std::string& id) {
        std::unique_lock<std::shared_timed_mutex> lk(mu, std::defer_lock);
        if (!lk.try_lock_for(std::chrono::seconds(10))) return;               // settings are written under mu
        if (id.empty()) settings.erase("embed_model"); else settings["embed_model"] = id;
        write_settings(a.settings, settings);
    };
    // A workspace file as it is, for the page's preview: text, or an image. ?path=src/main.cpp
    svr.Get("/api/workspace/file", [&](const httplib::Request& req, httplib::Response& res) {
        if (const std::string why = ws_check(req, false); !why.empty()) return oai_error(res, 403, why, "permission_error");
        const std::string folder = settings.value("workspace_folder", std::string()), rel = req.get_param_value("path");
        try {
            if (folder.empty()) throw workspace::Error("no workspace folder is chosen");
            const fs::path p = workspace::inside(fs::u8path(folder), rel);
            std::string ext = p.extension().u8string();
            for (auto& ch : ext) ch = (char)std::tolower((unsigned char)ch);
            static const std::map<std::string, std::string> images = {{".png", "image/png"}, {".jpg", "image/jpeg"}, {".jpeg", "image/jpeg"},
                {".gif", "image/gif"}, {".webp", "image/webp"}, {".bmp", "image/bmp"}, {".svg", "image/svg+xml"}};
            if (auto it = images.find(ext); it != images.end()) {
                std::error_code ec;
                if (!fs::is_regular_file(p, ec) || fs::file_size(p, ec) > 32u * 1048576) throw workspace::Error(rel + " is missing or larger than 32 MB");
                res.set_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'");   // (an SVG can hold scripts)
                return res.set_content(workspace::slurp(p, 32u * 1048576), it->second);
            }
            res.set_content(workspace::read_text(p, rel), "text/plain; charset=utf-8");
        } catch (const std::exception& e) { oai_error(res, 404, e.what()); }
    });

    // ---------------- saved chats, as files (tierlm's own page on this computer only) ----------------
    // Same checks as the workspace: these are your conversations. From another device the page keeps its chats in
    // that browser instead.
    svr.Get("/api/chats", [&](const httplib::Request& req, httplib::Response& res) {
        if (const std::string why = ws_check(req, false); !why.empty()) return oai_error(res, 403, why, "permission_error");
        res.set_content(dump({{"folder", chats_dir.u8string()}, {"chats", chats::list(chats_dir)}}), "application/json; charset=utf-8");
    });
    svr.Get(R"(/api/chats/([A-Za-z0-9]+))", [&](const httplib::Request& req, httplib::Response& res) {
        if (const std::string why = ws_check(req, false); !why.empty()) return oai_error(res, 403, why, "permission_error");
        try { res.set_content(dump(chats::load(chats_dir, req.matches[1])), "application/json; charset=utf-8"); }
        catch (const std::exception& e) { oai_error(res, 404, e.what()); }
    });
    svr.Post(R"(/api/chats/([A-Za-z0-9]+))", [&](const httplib::Request& req, httplib::Response& res) {
        if (const std::string why = ws_check(req, true); !why.empty()) return oai_error(res, 403, why, "permission_error");
        json c = json::parse(req.body, nullptr, false);
        if (!c.is_object()) return oai_error(res, 400, "expected a chat object");
        try {
            // (links to workspace files: the chat's own folder, else the one chosen now)
            const std::string wsf = c.is_object() && c.contains("workspace") && c["workspace"].is_string() ? c["workspace"].get<std::string>()
                                                                                                            : settings.value("workspace_folder", std::string());
            const fs::path f = chats::save(chats_dir, req.matches[1], c, wsf);
            res.set_content(dump({{"ok", true}, {"folder", f.filename().u8string()}}), "application/json; charset=utf-8");
        } catch (const std::exception& e) { oai_error(res, 500, e.what()); }
    });
    svr.Delete(R"(/api/chats/([A-Za-z0-9]+))", [&](const httplib::Request& req, httplib::Response& res) {
        if (const std::string why = ws_check(req, true); !why.empty()) return oai_error(res, 403, why, "permission_error");
        try { chats::remove(chats_dir, req.matches[1]); res.set_content("{\"ok\":true}", "application/json; charset=utf-8"); }
        catch (const std::exception& e) { oai_error(res, 500, e.what()); }
    });
    // An attached file, kept with the chat (so its link in chat.md works): {"name": "notes.txt", "text": "..."}.
    svr.Post(R"(/api/chats/([A-Za-z0-9]+)/files)", [&](const httplib::Request& req, httplib::Response& res) {
        if (const std::string why = ws_check(req, true); !why.empty()) return oai_error(res, 403, why, "permission_error");
        json b = json::parse(req.body, nullptr, false);
        if (!b.is_object() || !b.contains("name") || !b["text"].is_string()) return oai_error(res, 400, "expected {\"name\", \"text\"}");
        try { chats::put_file(chats_dir, req.matches[1], b.value("name", std::string()), b["text"].get<std::string>()); res.set_content("{\"ok\":true}", "application/json; charset=utf-8"); }
        catch (const std::exception& e) { oai_error(res, 500, e.what()); }
    });
    svr.Get(R"(/api/chats/([A-Za-z0-9]+)/files/(.+))", [&](const httplib::Request& req, httplib::Response& res) {
        if (const std::string why = ws_check(req, false); !why.empty()) return oai_error(res, 403, why, "permission_error");
        bool found = false;
        const std::string data = chats::get_file(chats_dir, req.matches[1], std::string(req.matches[2]), found);
        if (!found) return oai_error(res, 404, "no such file");
        res.set_header("Content-Disposition", "inline");
        res.set_content(data, "text/plain; charset=utf-8");
    });
    // Opens the chats folder (or one chat's folder) in Explorer, on this computer.
    svr.Post("/api/chats-folder", [&](const httplib::Request& req, httplib::Response& res) {
        if (const std::string why = ws_check(req, true); !why.empty()) return oai_error(res, 403, why, "permission_error");
        json b = json::parse(req.body.empty() ? "{}" : req.body, nullptr, false);
        fs::path p = chats_dir;
        if (b.is_object() && b.contains("id")) { const fs::path f = chats::folder_of(chats_dir, b.value("id", std::string())); if (!f.empty()) p = f; }
        std::error_code ec;
        fs::create_directories(p, ec);
#ifdef _WIN32
        ShellExecuteW(nullptr, L"open", p.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
        res.set_content(dump({{"opened", true}, {"folder", p.u8string()}}), "application/json; charset=utf-8");
#else
        res.set_content(dump({{"opened", false}, {"folder", p.u8string()}}), "application/json; charset=utf-8");
#endif
    });

    svr.Get("/api/embed", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(emb.mu);
        res.set_content(dump(embed_info()), "application/json; charset=utf-8");
    });
    svr.Post("/api/embed/load", [&](const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body, nullptr, false);
        const std::string id = body.is_object() ? body.value("model", std::string()) : std::string();
        auto file = model_file(id);
        if (file.empty()) file = bundled_file(id);
        if (file.empty()) { res.status = 400; res.set_content(dump({{"error", "model not found in the models folder"}}), "application/json; charset=utf-8"); return; }
        const json d = describe_model(file.u8string());
        if (!d.value("embedding", false)) {
            res.status = 400;
            res.set_content(dump({{"error", d.value("kind", std::string()) == "embedding" ? d.value("reason", std::string()) : "that is not an embedding model"}}), "application/json; charset=utf-8");
            return;
        }
        json info;
        try {
            std::lock_guard<std::mutex> lk(emb.mu);
            embed_load(file, id);
            info = embed_info();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "embedding model: %s\n", e.what());
            res.status = 500; res.set_content(dump({{"error", e.what()}}), "application/json; charset=utf-8"); return;
        }
        save_embed_choice(id);
        res.set_content(dump(info), "application/json; charset=utf-8");
    });
    // One click for an embedding model: nomic-embed-text v1.5 (Apache 2.0, about 140 MB), downloaded from Nomic's own
    // Hugging Face page into the models folder and checked (size and SHA-256, as Hugging Face states them), then
    // the page loads it. POST starts it (this computer only); GET says how it's going.
    static download::Status dl;
    static std::mutex dl_mu;
    static const char* kNomicUrl = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf";
    static const char* kNomicFile = "nomic-embed-text-v1.5.Q8_0.gguf";
    svr.Get("/api/embed/download", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(dl_mu);
        long long bytes = dl.bytes;
        if (dl.state == "downloading") {
            std::error_code ec;
            const auto n = fs::file_size(own_models / (std::string(kNomicFile) + ".part"), ec);
            if (!ec) bytes = (long long)n;
        }
        res.set_content(dump({{"state", dl.state}, {"bytes", bytes}, {"total", (long long)dl.total}, {"error", dl.error},
                              {"file", dl.file}, {"size_hint", 146 * 1000 * 1000}}), "application/json; charset=utf-8");
    });
    svr.Post("/api/embed/download", [&](const httplib::Request& req, httplib::Response& res) {
        if (!is_local(req)) { res.status = 403; res.set_content(dump({{"error", "only the computer running tierlm can download models"}}), "application/json; charset=utf-8"); return; }
        {
            std::lock_guard<std::mutex> lk(dl_mu);
            if (dl.running) { res.set_content(dump({{"state", dl.state}}), "application/json; charset=utf-8"); return; }
            dl.running = true; dl.state = "downloading"; dl.error.clear(); dl.file.clear(); dl.bytes = 0; dl.total = 0;
        }
        const fs::path dir = own_models;                                 // downloads: always tierlm's own folder
        // Already somewhere tierlm looks (its own folder, the chosen one, or LM Studio's built-ins): use that one
        // instead of downloading another copy.
        {
            bool exists = false;
            std::string found;
            for (const auto& m : scan_models(exists)) {
                std::string f = m.value("file", std::string()), low = f;
                for (char& ch : low) ch = (char)std::tolower((unsigned char)ch);
                if (low.rfind("nomic-embed-text-v1.5", 0) == 0 && m.value("embedding", false)) { found = m.value("id", std::string()); if (m.value("own", false)) break; }
            }
            if (!found.empty()) {
                std::lock_guard<std::mutex> lk(dl_mu);
                dl.running = false; dl.state = "done"; dl.file = found; dl.bytes = dl.total = 0;
                std::fprintf(stderr, "download: nomic-embed-text v1.5 is already here (%s); not downloading another copy\n", found.c_str());
                res.set_content(dump({{"state", "done"}, {"file", found}}), "application/json; charset=utf-8");
                return;
            }
        }
        const std::string url = std::getenv("TIERLM_NOMIC_URL") ? std::getenv("TIERLM_NOMIC_URL") : kNomicUrl;   // (tests)
        const std::string nomic_id = model_key((dir / kNomicFile).u8string());   // its id once saved
        std::thread([dir, url, nomic_id] {
            auto fail = [](const std::string& why) {
                std::lock_guard<std::mutex> lk(dl_mu);
                dl.state = "error"; dl.error = why; dl.running = false;
                std::fprintf(stderr, "download: %s\n", why.c_str());
            };
            std::error_code ec;
            fs::create_directories(dir, ec);
            const fs::path dest = dir / kNomicFile, part = dir / (std::string(kNomicFile) + ".part"), hdr = dir / (std::string(kNomicFile) + ".headers");
            const std::string curl = download::curl_exe();
            // 1. what the file should be: Hugging Face's first answer has its SHA-256 and size
            if (download::run({curl, "-sS", "-I", "--max-time", "30", "-o", hdr.u8string(), url}) != 0) {
                fs::remove(hdr, ec);
                return fail("couldn't reach huggingface.co (is this PC online? curl is needed: Windows 10 and 11 have it)");
            }
            std::string headers;
            { std::ifstream f(hdr, std::ios::binary); headers.assign(std::istreambuf_iterator<char>(f), {}); }
            fs::remove(hdr, ec);
            std::string sha = download::header(headers, "x-linked-etag");
            const long long size = std::atoll(download::header(headers, "x-linked-size").c_str());
            for (char& ch : sha) ch = (char)std::tolower((unsigned char)ch);
            if (sha.size() != 64 || size <= 0) return fail("Hugging Face didn't say the file's checksum; try again later");
            { std::lock_guard<std::mutex> lk(dl_mu); dl.total = size; }
            // Already here and intact: nothing to download.
            if (fs::exists(dest, ec) && (long long)fs::file_size(dest, ec) == size && download::sha256_file(dest) == sha) {
                std::lock_guard<std::mutex> lk(dl_mu);
                dl.state = "done"; dl.file = nomic_id; dl.bytes = size; dl.running = false;
                return;
            }
            // 2. the file, to a .part name
            std::fprintf(stderr, "download: %s (%.0f MB)\n", url.c_str(), size / 1e6);
            if (download::run({curl, "-sS", "-f", "-L", "--retry", "3", "-o", part.u8string(), url}) != 0) {
                fs::remove(part, ec);
                return fail("the download failed (connection lost?); try again");
            }
            // 3. checked, then given its name
            { std::lock_guard<std::mutex> lk(dl_mu); dl.state = "checking"; dl.bytes = size; }
            if ((long long)fs::file_size(part, ec) != size || download::sha256_file(part) != sha) {
                fs::remove(part, ec);
                return fail("the downloaded file didn't match Hugging Face's checksum, so it was deleted; try again");
            }
            fs::rename(part, dest, ec);
            if (ec) { fs::remove(part, ec); return fail("couldn't save " + dest.u8string() + ": " + ec.message()); }
            std::lock_guard<std::mutex> lk(dl_mu);
            dl.state = "done"; dl.file = nomic_id; dl.running = false;
            std::fprintf(stderr, "download: %s saved and checked (SHA-256 %s)\n", dest.u8string().c_str(), sha.c_str());
        }).detach();
        res.set_content(dump({{"state", "downloading"}}), "application/json; charset=utf-8");
    });
    // ---- Get models: search Hugging Face for GGUF models and download them ----
    // Everything goes through curl (as the embedding model's download), to Hugging Face's public API: no account is
    // needed for public models. Search lists GGUF repos; a repo's files are grouped into the model's quants (split
    // files together), vision files (mmproj) and draft files (mtp-); the first model file's header is read (just
    // its first megabytes) to say what it is and whether this tierlm runs it. A download goes into the models folder,
    // in <owner>/<repo>/, each file to a .part name that resumes after an interruption, checked against Hugging
    // Face's SHA-256 and size, then given its name. One download at a time. Only on the computer running tierlm.
    // TIERLM_HF_BASE (tests): another address in place of https://huggingface.co.
    static const std::string hf_base = std::getenv("TIERLM_HF_BASE") ? std::getenv("TIERLM_HF_BASE") : "https://huggingface.co";
    auto url_enc = [](const std::string& v) {
        std::string o; char b[4];
        for (unsigned char ch : v) {
            if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.' || ch == '~') o += (char)ch;
            else { std::snprintf(b, sizeof b, "%%%02X", ch); o += b; }
        }
        return o;
    };
    auto path_enc = [url_enc](const std::string& p) {             // a repo file's path, its slashes kept
        std::string o; size_t at = 0;
        while (at <= p.size()) { size_t e = p.find('/', at); if (e == std::string::npos) e = p.size(); o += (o.empty() && at == 0 ? "" : "/") + url_enc(p.substr(at, e - at)); at = e + 1; }
        return o;
    };
    auto valid_repo = [](const std::string& id) {                // owner/name, nothing that could leave the API
        if (id.empty() || id.size() > 200 || std::count(id.begin(), id.end(), '/') != 1 || id.find("..") != std::string::npos) return false;
        for (unsigned char ch : id) if (!(std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.' || ch == '/')) return false;
        return true;
    };
    // curl to a temporary file; the body (or "" with *code set on failure)
    auto hf_get = [](const std::string& url, int* code, const std::string& range = std::string()) {
        std::error_code ec;
        static std::atomic<int> seq{0};
        const fs::path tmp = fs::temp_directory_path(ec) / ("tierlm-hf-" + std::to_string(++seq) + ".tmp");
        std::vector<std::string> args = {download::curl_exe(), "-sS", "-f", "-L", "--max-time", "60", "-o", tmp.u8string()};
        if (!range.empty()) { args.push_back("-r"); args.push_back(range); }
        args.push_back(url);
        const int rc = download::run(args);
        std::string body;
        { std::ifstream f(tmp, std::ios::binary); body.assign(std::istreambuf_iterator<char>(f), {}); }
        fs::remove(tmp, ec);
        if (code) *code = rc;
        return body;
    };
    svr.Get("/api/hf/search", [&](const httplib::Request& req, httplib::Response& res) {
        if (!is_local(req)) return oai_error(res, 403, "only on the computer running tierlm");
        const std::string q = req.get_param_value("q");
        std::string sort = req.get_param_value("sort");
        if (sort != "likes" && sort != "lastModified" && sort != "trendingScore") sort = "downloads";
        int rc = 0;
        const std::string body = hf_get(hf_base + "/api/models?filter=gguf&limit=60&sort=" + sort + "&direction=-1" +
                                        (q.empty() ? std::string() : "&search=" + url_enc(q)), &rc);
        const json j = json::parse(body, nullptr, false);
        if (rc != 0 || !j.is_array()) return oai_error(res, 502, rc != 0 ? "couldn't reach huggingface.co (is this PC online?)" : "Hugging Face's answer wasn't understood");
        json out = json::array();
        for (const auto& m : j) {
            if (!m.is_object()) continue;
            out.push_back({{"id", m.value("id", std::string())}, {"downloads", m.value("downloads", 0LL)}, {"likes", m.value("likes", 0LL)},
                           {"updated", m.value("lastModified", m.value("createdAt", std::string()))}});
        }
        res.set_content(dump(json{{"models", out}}), "application/json; charset=utf-8");
    });
    // The quant's name in a file name: UD-Q4_K_XL, Q4_K_M, IQ4_XS, Q8_0, BF16, MXFP4...
    auto quant_of = [](const std::string& name) {
        static const std::regex re(R"(((?:UD-)?(?:I?Q\d_[A-Z0-9_]+|Q\d_\d|BF16|F16|F32|MXFP4[A-Z0-9_]*|TQ\d_\d)))", std::regex::icase);
        std::smatch m;
        return std::regex_search(name, m, re) ? m.str(1) : fs::u8path(name).stem().u8string();
    };
    svr.Get("/api/hf/repo", [&](const httplib::Request& req, httplib::Response& res) {
        if (!is_local(req)) return oai_error(res, 403, "only on the computer running tierlm");
        const std::string id = req.get_param_value("id");
        if (!valid_repo(id)) return oai_error(res, 400, "not a Hugging Face model name (owner/name)");
        int rc = 0;
        const json j = json::parse(hf_get(hf_base + "/api/models/" + id + "?blobs=true", &rc), nullptr, false);
        if (rc != 0 || !j.is_object() || !j.contains("siblings")) return oai_error(res, 502, rc != 0 ? "couldn't reach huggingface.co" : "Hugging Face didn't list that model's files");
        // group the GGUF files
        static const std::regex part_re(R"((.*)-(\d{5})-of-(\d{5})\.gguf$)", std::regex::icase);
        std::map<std::string, json> groups;                      // key -> {kind, label, files, size}
        for (const auto& f : j["siblings"]) {
            const std::string path = f.value("rfilename", std::string());
            if (path.size() < 5 || path.compare(path.size() - 5, 5, ".gguf") != 0) continue;
            const std::string name = fs::u8path(path).filename().u8string();
            std::string lower = name; for (char& ch : lower) ch = (char)std::tolower((unsigned char)ch);
            const std::string kind = lower.rfind("mmproj", 0) == 0 ? "vision" : lower.rfind("mtp-", 0) == 0 ? "draft" : "model";
            std::smatch m;
            const std::string key = std::regex_match(path, m, part_re) ? m.str(1) : path.substr(0, path.size() - 5);
            const long long size = f.contains("lfs") && f["lfs"].is_object() ? f["lfs"].value("size", f.value("size", 0LL)) : f.value("size", 0LL);
            const std::string sha = f.contains("lfs") && f["lfs"].is_object() ? f["lfs"].value("sha256", std::string()) : std::string();
            json& g = groups[kind + "\n" + key];
            if (g.is_null()) g = {{"kind", kind}, {"label", quant_of(name)}, {"files", json::array()}, {"size", 0LL}};
            g["files"].push_back({{"path", path}, {"size", size}, {"sha256", sha}});
            g["size"] = g["size"].get<long long>() + size;
        }
        json quants = json::array(), vision = json::array(), draft = json::array();
        for (auto& [k, g] : groups) {
            auto& fl = g["files"];
            std::sort(fl.begin(), fl.end(), [](const json& a, const json& b) { return a["path"].get<std::string>() < b["path"].get<std::string>(); });
            (g["kind"] == "vision" ? vision : g["kind"] == "draft" ? draft : quants).push_back(g);
        }
        std::sort(quants.begin(), quants.end(), [](const json& a, const json& b) { return a["size"].get<long long>() < b["size"].get<long long>(); });
        json out = {{"id", id}, {"quants", quants}, {"vision", vision}, {"draft", draft},
                    {"gated", j.contains("gated") && !(j["gated"].is_boolean() && !j["gated"].get<bool>())},
                    {"downloads", j.value("downloads", 0LL)}, {"likes", j.value("likes", 0LL)}};
        if (j.contains("gguf") && j["gguf"].is_object()) {
            out["arch"] = j["gguf"].value("architecture", std::string());
            out["ctx_train"] = j["gguf"].value("context_length", 0LL);
        }
        // what it is, from the first model file's header (its first megabytes)
        if (!quants.empty()) {
            const std::string first = quants[0]["files"][0]["path"];
            std::error_code ec;
            const fs::path tmp = fs::temp_directory_path(ec) / "tierlm-hf-header.gguf";
            for (const char* range : {"0-8388607", "0-67108863"}) {
                int rc2 = 0;
                const std::string head = hf_get(hf_base + "/" + id + "/resolve/main/" + path_enc(first), &rc2, range);
                if (rc2 != 0 || head.size() < 16 || head.compare(0, 4, "GGUF") != 0) break;
                { std::ofstream o(tmp, std::ios::binary); o.write(head.data(), (std::streamsize)head.size()); }
                gguf_init_params ip{ true, nullptr };
                gguf_context* g = gguf_init_from_file(tmp.u8string().c_str(), ip);
                if (!g) continue;                                 // the header is longer: read more
                gguf_free(g);
                json d = describe_model(tmp.u8string(), true);
                for (const char* k : {"arch", "supported", "reason", "kind", "kv_bytes_per_token", "layers", "experts", "experts_used", "ctx_train", "hybrid", "name"})
                    if (d.contains(k)) out[k] = d[k];
                break;
            }
            fs::remove(tmp, ec);
        }
        if (!out.contains("supported") && out.contains("arch")) out["supported"] = arch_supported(out["arch"].get<std::string>());
        res.set_content(dump(out), "application/json; charset=utf-8");
    });
    struct HfFile { std::string path, sha; long long size = 0; std::string state = "waiting"; };
    struct HfJob {
        std::atomic<bool> running{false}, cancel{false};
        std::string state = "idle", error, id, folder;              // (under hf_mu)
        std::vector<HfFile> files;
        int cur = -1;
    };
    static HfJob hf;
    static std::mutex hf_mu;
    svr.Get("/api/hf/download", [&](const httplib::Request& req, httplib::Response& res) {
        if (!is_local(req)) return oai_error(res, 403, "only on the computer running tierlm");
        std::lock_guard<std::mutex> lk(hf_mu);
        json files = json::array();
        long long done = 0, total = 0;
        for (size_t i = 0; i < hf.files.size(); ++i) {
            const HfFile& f = hf.files[i];
            long long have = f.state == "done" ? f.size : 0;
            if (f.state == "downloading") {
                std::error_code ec;
                const auto n = fs::file_size(fs::u8path(hf.folder) / fs::u8path(f.path + ".part"), ec);
                if (!ec) have = (long long)n;
            }
            if (f.state == "checking") have = f.size;
            done += have; total += f.size;
            files.push_back({{"path", f.path}, {"size", f.size}, {"done", have}, {"state", f.state}});
        }
        res.set_content(dump(json{{"state", hf.state}, {"id", hf.id}, {"error", hf.error}, {"files", files}, {"done", done}, {"total", total},
                                  {"folder", hf.folder}}), "application/json; charset=utf-8");
    });
    svr.Post("/api/hf/cancel", [&](const httplib::Request&, httplib::Response& res) {
        hf.cancel = true;
        res.set_content(dump(json{{"ok", true}}), "application/json; charset=utf-8");
    });
    svr.Post("/api/hf/download", [&](const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body, nullptr, false);
        if (!body.is_object() || !body.contains("files") || !body["files"].is_array()) return oai_error(res, 400, "expected {\"id\": \"owner/name\", \"files\": [paths]}");
        const std::string id = body.value("id", std::string());
        if (!valid_repo(id)) return oai_error(res, 400, "not a Hugging Face model name (owner/name)");
        if (hf.running) return oai_error(res, 409, "a download is already running; wait for it or cancel it");
        // the files, as Hugging Face lists them (sizes and checksums from there, not from the request)
        int rc = 0;
        const json j = json::parse(hf_get(hf_base + "/api/models/" + id + "?blobs=true", &rc), nullptr, false);
        if (rc != 0 || !j.is_object() || !j.contains("siblings")) return oai_error(res, 502, "couldn't reach huggingface.co");
        std::vector<HfFile> want;
        for (const auto& p : body["files"]) {
            if (!p.is_string()) continue;
            const std::string path = p.get<std::string>();
            if (path.find("..") != std::string::npos || path.empty() || path[0] == '/' || path.find('\\') != std::string::npos) return oai_error(res, 400, "bad file name");
            for (const auto& f : j["siblings"])
                if (f.value("rfilename", std::string()) == path) {
                    HfFile hfile; hfile.path = path;
                    hfile.size = f.contains("lfs") && f["lfs"].is_object() ? f["lfs"].value("size", f.value("size", 0LL)) : f.value("size", 0LL);
                    hfile.sha = f.contains("lfs") && f["lfs"].is_object() ? f["lfs"].value("sha256", std::string()) : std::string();
                    want.push_back(hfile);
                }
        }
        if (want.empty()) return oai_error(res, 400, "none of those files are in " + id);
        const fs::path folder = own_models / fs::u8path(id);             // always tierlm's own folder
        long long need = 0;
        for (const auto& f : want) need += f.size;
        {
            std::error_code ec;
            fs::create_directories(folder, ec);
            unsigned long long fr = 0, tot = 0;
            if (drive_space(folder.u8string(), fr, tot) && (long long)fr < need + (1LL << 30))
                return oai_error(res, 507, "not enough room on the models folder's drive: needs " + std::to_string((need >> 30) + 1) + " GB, has " + std::to_string(fr >> 30) + " GB");
        }
        {
            std::lock_guard<std::mutex> lk(hf_mu);
            hf.running = true; hf.cancel = false; hf.state = "downloading"; hf.error.clear(); hf.id = id; hf.folder = folder.u8string(); hf.files = want; hf.cur = 0;
        }
        std::fprintf(stderr, "get models: %s, %zu files, %.1f GB, into %s\n", id.c_str(), want.size(), need / 1e9, folder.u8string().c_str());
        std::thread([id, folder, path_enc] {
            auto set = [](size_t i, const char* st) { std::lock_guard<std::mutex> lk(hf_mu); hf.files[i].state = st; };
            auto finish = [](const std::string& state, const std::string& why) {
                std::lock_guard<std::mutex> lk(hf_mu);
                hf.state = state; hf.error = why; hf.running = false;
                std::fprintf(stderr, "get models: %s%s%s\n", state.c_str(), why.empty() ? "" : ": ", why.c_str());
            };
            const std::string curl = download::curl_exe();
            std::vector<HfFile> files;
            { std::lock_guard<std::mutex> lk(hf_mu); files = hf.files; }
            for (size_t i = 0; i < files.size(); ++i) {
                const HfFile& f = files[i];
                std::error_code ec;
                const fs::path dest = folder / fs::u8path(f.path), part = folder / fs::u8path(f.path + ".part");
                fs::create_directories(dest.parent_path(), ec);
                { std::lock_guard<std::mutex> lk(hf_mu); hf.cur = (int)i; }
                auto good = [&](const fs::path& p) {
                    if ((long long)fs::file_size(p, ec) != f.size || ec) return false;
                    if (f.sha.size() != 64) return true;
                    set(i, "checking");
                    return download::sha256_file(p) == f.sha;
                };
                if (fs::exists(dest, ec) && good(dest)) { set(i, "done"); continue; }     // already here and intact
                set(i, "downloading");
                const std::string url = hf_base + "/" + id + "/resolve/main/" + path_enc(f.path);
                const int rc = download::run({curl, "-sS", "-f", "-L", "--retry", "5", "--retry-delay", "3", "-C", "-", "-o", part.u8string(), url}, &hf.cancel);
                if (rc == -2) { set(i, "cancelled"); return finish("cancelled", "stopped; what was downloaded is kept and continues next time"); }
                if (rc != 0) {
                    set(i, "error");
                    if ((long long)fs::file_size(part, ec) > f.size) fs::remove(part, ec);
                    return finish("error", rc == 22 ? "Hugging Face refused " + f.path + " (a gated model needs its license accepted and a token, which tierlm doesn't do yet)"
                                                    : "the download of " + f.path + " stopped (connection lost?); start it again to continue where it left off");
                }
                if (!good(part)) { fs::remove(part, ec); set(i, "error"); return finish("error", f.path + " didn't match Hugging Face's checksum, so it was deleted; try again"); }
                fs::rename(part, dest, ec);
                if (ec) { set(i, "error"); return finish("error", "couldn't save " + dest.u8string() + ": " + ec.message()); }
                set(i, "done");
                std::fprintf(stderr, "get models: %s saved and checked\n", dest.u8string().c_str());
            }
            finish("done", "");
        }).detach();
        res.set_content(dump(json{{"ok", true}, {"folder", folder.u8string()}}), "application/json; charset=utf-8");
    });

    svr.Post("/api/embed/unload", [&](const httplib::Request&, httplib::Response& res) {
        { std::lock_guard<std::mutex> lk(emb.mu); emb.enc.reset(); emb.id.clear(); emb.path.clear(); emb.name.clear(); }
        save_embed_choice("");
        std::fprintf(stderr, "embedding model: unloaded\n");
        res.set_content(dump({{"loaded", false}}), "application/json; charset=utf-8");
    });

    // POST /v1/embeddings, as OpenAI's: {"input": "text" | ["text", ...] | [token ids] | [[token ids], ...],
    // "encoding_format": "float" | "base64", "dimensions": n}. Each vector is normalised to length 1, as OpenAI's
    // are; "dimensions" keeps the first n numbers (for models trained for it, such as nomic-embed-text v1.5) and
    // normalises again. Texts longer than the model takes are cut to fit (as sentence-transformers does).
    svr.Post(R"(/(?:v1|api/v0)/embeddings)", [&](const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object() || !body.contains("input")) return oai_error(res, 400, "expected a JSON body with \"input\"");
        std::vector<std::vector<int32_t>> inputs;
        std::vector<std::string> texts;
        const json& in = body["input"];
        auto is_ids = [](const json& v) { return v.is_array() && !v.empty() && std::all_of(v.begin(), v.end(), [](const json& x) { return x.is_number_integer(); }); };
        std::lock_guard<std::mutex> lk(emb.mu);
        if (!emb.enc) return oai_error(res, 400, "no embedding model is loaded: in tierlm's side menu, open Model info and choose an Embedding model");
        try {
            if (in.is_string()) texts.push_back(in.get<std::string>());
            else if (is_ids(in)) inputs.push_back(in.get<std::vector<int32_t>>());
            else if (in.is_array()) {
                for (const auto& x : in) {
                    if (x.is_string()) texts.push_back(x.get<std::string>());
                    else if (is_ids(x)) inputs.push_back(x.get<std::vector<int32_t>>());
                    else return oai_error(res, 400, "\"input\" must be a string, a list of strings, or token ids");
                }
            } else return oai_error(res, 400, "\"input\" must be a string, a list of strings, or token ids");
            if (!texts.empty() && !inputs.empty()) return oai_error(res, 400, "\"input\" mixes texts and token ids");
            if (texts.empty() && inputs.empty()) return oai_error(res, 400, "\"input\" is empty");
            const int maxn = emb.enc->max_tokens();
            for (const std::string& t : texts) {                              // [CLS] text [SEP], cut to fit
                std::vector<int32_t> ids, body_ids = emb.tok.encode(t);
                const int room = maxn - (emb.tok.cls() >= 0) - (emb.tok.sep() >= 0);
                if ((int)body_ids.size() > room) body_ids.resize(std::max(0, room));
                if (emb.tok.cls() >= 0) ids.push_back(emb.tok.cls());
                ids.insert(ids.end(), body_ids.begin(), body_ids.end());
                if (emb.tok.sep() >= 0) ids.push_back(emb.tok.sep());
                inputs.push_back(std::move(ids));
            }
            const int dims = body.value("dimensions", 0);
            const bool b64 = body.value("encoding_format", std::string("float")) == "base64";
            const int vocab = (int)emb.tok.vocab_size();
            json data = json::array();
            size_t n_tok = 0;
            const auto t0 = Clock::now();
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto& ids = inputs[i];
                if (ids.empty()) return oai_error(res, 400, "input " + std::to_string(i) + " is empty");
                for (int32_t t : ids) if (t < 0 || t >= vocab) return oai_error(res, 400, "token id out of range in input " + std::to_string(i));
                if ((int)ids.size() > maxn) ids.resize(maxn);
                n_tok += ids.size();
                std::vector<float> v = emb.enc->embed(ids, eng.threads);
                if (dims > 0 && dims < (int)v.size()) v.resize(dims);
                double norm = 0; for (float x : v) norm += (double)x * x;
                norm = std::sqrt(norm);
                if (norm > 0) for (float& x : v) x = (float)(x / norm);
                json item = {{"object", "embedding"}, {"index", i}};
                if (b64) item["embedding"] = base64_encode(reinterpret_cast<const unsigned char*>(v.data()), v.size() * sizeof(float));
                else item["embedding"] = v;
                data.push_back(std::move(item));
            }
            std::fprintf(stderr, "embeddings: %zu input%s, %zu tokens in %.2f s\n", inputs.size(), inputs.size() == 1 ? "" : "s", n_tok, since(t0));
            res.set_content(dump({{"object", "list"}, {"data", data}, {"model", emb.id},
                                  {"usage", {{"prompt_tokens", n_tok}, {"total_tokens", n_tok}}}}), "application/json; charset=utf-8");
        } catch (const std::exception& e) {
            return oai_error(res, 500, e.what(), "server_error");
        }
    });
    // Load the embedding model chosen last time.
    if (settings.contains("embed_model") && settings["embed_model"].is_string()) {
        const std::string id = settings["embed_model"];
        auto file = model_file(id);
        if (file.empty()) file = bundled_file(id);
        if (!file.empty()) {
            try { std::lock_guard<std::mutex> lk(emb.mu); embed_load(file, id); }
            catch (const std::exception& e) { std::fprintf(stderr, "embedding model %s: %s\n", id.c_str(), e.what()); }
        }
    }

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
