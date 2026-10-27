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
//                       "flash_attn": true, "kv_type": "f16|f32", "batch": 256, "experts_used": 0, "seed": -1}
//   POST /api/unload   frees the model
//   POST /api/chat    {"messages":[{"role":"user","content":"..."}], "temperature":0.7, "top_p":0.9, "max_tokens":512}
//                     streams Server-Sent Events: one "prompt" event, a "token" event per token
//                     (with the experts picked in every layer), then "done"
//   POST /api/reset   clears the KV cache
#include "olmoe.h"
#include "tokenizer.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "httplib.h"
#include "json.hpp"

#include <algorithm>
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

#ifdef _WIN32
#include <windows.h>
#include <shobjidl.h>
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
    bool models_given = false;
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

// OLMoE-Instruct (Tulu) chat format. Control tokens are matched when encoding with parse_special.
std::string format_chat(const json& messages) {
    std::string s = "<|endoftext|>";
    for (const auto& m : messages) {
        const std::string role = m.value("role", ""), content = m.value("content", "");
        if (role == "system")         s += "<|system|>\n" + content + "\n";
        else if (role == "user")      s += "<|user|>\n" + content + "\n";
        else if (role == "assistant") s += "<|assistant|>\n" + content + "<|endoftext|>\n";
    }
    return s + "<|assistant|>\n";
}

// Temperature + nucleus (top-p) sampling. temperature <= 0 means always take the most likely token.
int32_t sample(const std::vector<float>& lg, float temperature, float top_p, std::mt19937& rng) {
    const int32_t best = (int32_t)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    if (temperature <= 0.0f) return best;
    const float mx = lg[best];
    std::vector<std::pair<float, int32_t>> p(lg.size());
    double total = 0;
    for (size_t i = 0; i < lg.size(); ++i) {
        float w = std::exp((lg[i] - mx) / temperature);
        p[i] = {w, (int32_t)i};
        total += w;
    }
    const size_t keep = std::min<size_t>(256, p.size());     // the nucleus is always inside the top few hundred
    std::partial_sort(p.begin(), p.begin() + keep, p.end(), [](auto& a, auto& b) { return a.first > b.first; });
    double cum = 0;
    size_t n = 0;
    while (n < keep && cum < top_p * total) cum += p[n++].first;
    std::uniform_real_distribution<double> u(0.0, cum);
    double r = u(rng);
    for (size_t i = 0; i < n; ++i) { r -= p[i].first; if (r <= 0) return p[i].second; }
    return p[n ? n - 1 : 0].second;
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
    std::unique_ptr<OlmoeModel> model;
    Tokenizer tok;
    std::vector<int32_t> cached;      // the tokens currently in the KV cache
    std::string path, name;
    int threads = 1, seed = -1;
    double load_s = 0;
    std::mt19937 rng{std::random_device{}()};
};

// Header-only look at a GGUF file: what it is, how big, and whether tierlm can run it.
json describe_model(const std::string& path) {
    json j = {{"file", std::filesystem::u8path(path).filename().u8string()}, {"supported", false}};
    std::error_code ec;
    j["bytes"] = (uint64_t)std::filesystem::file_size(std::filesystem::u8path(path), ec);
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
    const int64_t layers = u32(arch + ".block_count"), embd = u32(arch + ".embedding_length");
    const int64_t heads = u32(arch + ".attention.head_count"), kv_heads = u32(arch + ".attention.head_count_kv");
    j["layers"] = layers;
    j["experts"] = u32(arch + ".expert_count");
    j["experts_used"] = u32(arch + ".expert_used_count");
    j["ctx_train"] = u32(arch + ".context_length");
    // KV cache cost per token of context, at f16: K and V for every layer.
    if (layers && embd && heads) j["kv_bytes_per_token"] = 2 * layers * (kv_heads ? kv_heads : heads) * (embd / heads) * 2;
    if (arch == "olmoe") j["supported"] = true;
    else j["reason"] = "the " + (arch.empty() ? std::string("unknown") : arch) + " architecture is not supported yet";
    gguf_free(g);
    return j;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (k == "--port") a.port = std::atoi(next().c_str());
        else if (k == "--host") a.host = next();
        else if (k == "-d") a.device = next();
        else if (k == "-t") a.threads = std::atoi(next().c_str());
        else if (k == "-c") a.ctx = std::atoi(next().c_str());
        else if (k == "--web") a.web = next();
        else if (k == "--models") { a.models = next(); a.models_given = true; }
        else if (k == "--settings") a.settings = next();
        else if (k == "-h" || k == "--help") {
            std::fprintf(stderr, "usage: %s [model.gguf] [--port 8080] [--host 127.0.0.1] [-d auto|cpu|gpu|<device>] [-t threads] [-c context] [--web dir] [--models dir] [--settings file]\n"
                                 "Without a model, the web page opens a setup screen to choose the hardware and model.\n", argv[0]);
            return 0;
        }
        else if (a.model.empty()) a.model = k;
    }

    Engine eng;
    eng.threads = a.threads;
    std::mutex mu;                       // one load or generation at a time

    // The models folder: --models if given, else the one saved last time, else "models".
    json settings = read_settings(a.settings);
    bool recursive = settings.value("recursive", true);
    if (!a.models_given && settings.contains("models_folder") && settings["models_folder"].is_string()) {
        std::error_code ec;
        const std::string saved = settings["models_folder"];
        if (fs::is_directory(fs::u8path(saved), ec)) a.models = saved;
    }
    std::fprintf(stderr, "tierlm %s\nmodels folder: %s\n", TIERLM_VERSION, a.models.c_str());

    auto load = [&](const std::string& path, const LoadOptions& opt, int threads, int seed) {
        json info = describe_model(path);
        if (!info.value("supported", false)) throw std::runtime_error(info.value("reason", std::string("unsupported model")));
        eng.model.reset();               // free the old model (and its GPU memory) first
        eng.cached.clear();
        auto t0 = Clock::now();
        eng.tok = Tokenizer::from_gguf(path);
        eng.model = OlmoeModel::load(path, opt);
        eng.path = path;
        eng.name = gguf_string(path, "general.name");
        eng.threads = threads > 0 ? threads : a.threads;
        eng.seed = seed;
        if (seed >= 0) eng.rng.seed((uint32_t)seed);
        eng.load_s = since(t0);
        std::fprintf(stderr, "loaded %s on %s in %.1fs\n", path.c_str(), eng.model->device_name().c_str(), eng.load_s);
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
        std::unique_lock<std::mutex> lk(mu, std::try_to_lock);
        if (!lk.owns_lock()) { json j = last_info; j["busy"] = true; res.set_content(dump(j), "application/json"); return; }
        if (!eng.model) { last_info = {{"loaded", false}, {"version", TIERLM_VERSION}}; res.set_content(dump(last_info), "application/json"); return; }
        const auto& hp = eng.model->hparams();
        size_t free = 0, total = 0;
        eng.model->device_memory(free, total);
        json j = {
            {"loaded", true}, {"path", eng.path},
            {"name", eng.name.empty() ? "OLMoE" : eng.name}, {"arch", "olmoe"},
            {"layers", hp.n_layer}, {"experts", hp.n_expert}, {"experts_used", hp.n_expert_used},
            {"vocab", hp.n_vocab}, {"ctx", eng.model->max_ctx()}, {"n_past", eng.model->n_past()},
            {"device", eng.model->device_name()}, {"on_gpu", eng.model->on_gpu()}, {"threads", eng.threads},
            {"device_type", eng.model->integrated_gpu() ? "igpu" : eng.model->on_gpu() ? "gpu" : "cpu"},
            {"weights_bytes", eng.model->weights_bytes()}, {"kv_bytes", eng.model->kv_bytes()},
            {"device_free", free}, {"device_total", total},
            {"kv_type", eng.model->kv_type()}, {"flash_attn", eng.model->flash_attn()}, {"load_s", eng.load_s},
            {"batch", eng.model->batch()}, {"experts_default", eng.model->experts_default()}, {"seed", eng.seed},
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
        json j = {{"devices", list}, {"threads", a.threads}, {"hw_threads", (int)std::thread::hardware_concurrency()},
                  {"warnings", hardware_warnings(have_gpu)}, {"version", TIERLM_VERSION}};
        res.set_content(dump(j), "application/json");
    });

    // Models in the models folder (and, if enabled, its subfolders, as LM Studio stores them), read from
    // their headers only (fast: no weights are loaded). "id" is the path relative to the folder.
    svr.Get("/api/models", [&](const httplib::Request&, httplib::Response& res) {
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
        const bool exists = fs::is_directory(base, ec);
        if (exists && recursive) {
            auto it = fs::recursive_directory_iterator(base, fs::directory_options::skip_permission_denied, ec);
            for (; !ec && it != fs::recursive_directory_iterator() && list.size() < 500; it.increment(ec)) {
                if (it.depth() > 4) { it.disable_recursion_pending(); continue; }
                std::error_code e2;
                if (it->is_regular_file(e2) && it->path().extension() == ".gguf") add(it->path());
            }
        } else if (exists) {
            for (const auto& f : fs::directory_iterator(base, ec))
                if (f.is_regular_file() && f.path().extension() == ".gguf") add(f.path());
        }
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
        std::lock_guard<std::mutex> lk(mu);
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

    svr.Post("/api/load", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("model")) {
            res.status = 400;
            res.set_content("{\"error\":\"expected {\\\"model\\\": \\\"...\\\"}\"}", "application/json");
            return;
        }
        if (!mu.try_lock()) {
            res.status = 409;
            res.set_content("{\"error\":\"busy: wait for the current reply or load to finish\"}", "application/json");
            return;
        }
        std::lock_guard<std::mutex> lk(mu, std::adopt_lock);
        // Only files inside the models folder can be loaded.
        std::error_code ec;
        const auto base = std::filesystem::weakly_canonical(std::filesystem::u8path(a.models), ec);
        const auto file = std::filesystem::weakly_canonical(std::filesystem::u8path(a.models) / std::filesystem::u8path(body.value("model", std::string())), ec);
        auto rel = file.lexically_relative(base);
        if (ec || rel.empty() || rel.begin()->string() == ".." || !std::filesystem::is_regular_file(file, ec)) {
            res.status = 400;
            res.set_content(dump({{"error", "model not found in the models folder"}}), "application/json");
            return;
        }
        try {
            LoadOptions o;
            o.device = body.value("device", std::string("auto"));
            o.max_ctx = body.value("ctx", a.ctx);
            if (body.contains("flash_attn")) o.flash_attn = body.value("flash_attn", true) ? 1 : 0;
            if (body.contains("kv_type")) o.kv_f32 = body.value("kv_type", std::string("f16")) == "f32" ? 1 : 0;
            o.batch = body.value("batch", 0);
            o.experts_used = body.value("experts_used", 0);
            const int threads = body.value("threads", a.threads), seed = body.value("seed", -1);
            load(file.u8string(), o, threads, seed);
            // Remember these settings for this model, like LM Studio does.
            const std::string id = body.value("model", std::string());
            json saved = body;
            saved.erase("model");
            settings["models"][id] = saved;
            settings["last_model"] = id;
            write_settings(a.settings, settings);
            res.set_content(dump({{"ok", true}, {"device", eng.model->device_name()}, {"load_s", eng.load_s}}), "application/json");
        } catch (const std::exception& e) {
            eng.model.reset();
            eng.path.clear();
            res.status = 500;
            res.set_content(dump({{"error", e.what()}}), "application/json");
        }
    });

    svr.Post("/api/unload", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(mu);
        eng.model.reset();
        eng.cached.clear();
        eng.path.clear();
        res.set_content("{\"ok\":true}", "application/json");
    });

    svr.Post("/api/reset", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(mu);
        if (eng.model) eng.model->reset();
        eng.cached.clear();
        res.set_content("{\"ok\":true}", "application/json");
    });

    svr.Post("/api/chat", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content("{\"error\":\"expected {\\\"messages\\\": [...]}\"}", "application/json");
            return;
        }
        if (!mu.try_lock()) {
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
        auto lock = std::make_shared<std::unique_lock<std::mutex>>(mu, std::adopt_lock);   // released when the stream ends
        res.set_header("Cache-Control", "no-cache");

        res.set_chunked_content_provider("text/event-stream", [&, body, lock](size_t, httplib::DataSink& sink) {
            auto send = [&](const json& j) { std::string s = "data: " + dump(j) + "\n\n"; return sink.write(s.data(), s.size()); };
            try {
                const float temperature = body.value("temperature", 0.7f);
                const float top_p = body.value("top_p", 0.9f);
                const int max_tokens = body.value("max_tokens", 512);
                if (eng.seed >= 0) eng.rng.seed((uint32_t)eng.seed);    // same seed, same reply
                auto& model = eng.model; auto& tok = eng.tok; auto& cached = eng.cached;
                const auto& hp = model->hparams();
                const int L = hp.n_layer, E = hp.n_expert, K = hp.n_expert_used;

                std::vector<int32_t> toks = tok.encode(format_chat(body["messages"]), true);
                if ((int)toks.size() + 1 >= model->max_ctx()) throw std::runtime_error("conversation is longer than the context window; start a new chat");

                // Reuse whatever part of the KV cache still matches this conversation.
                size_t common = 0;
                while (common < cached.size() && common < toks.size() && cached[common] == toks[common]) ++common;
                if (common == toks.size()) common = toks.size() - 1;      // must evaluate at least one token
                model->truncate((int)common);
                cached.resize(common);

                std::vector<int32_t> suffix(toks.begin() + common, toks.end()), routing;
                auto p0 = Clock::now();
                std::vector<float> lg = model->eval(suffix, eng.threads, &routing);
                const double prompt_s = since(p0);
                cached.insert(cached.end(), suffix.begin(), suffix.end());

                std::vector<int> counts((size_t)L * E, 0);                // experts used while reading the prompt
                for (size_t i = 0; i < routing.size(); ++i) counts[((i / K) % L) * E + routing[i]]++;
                send({{"type", "prompt"}, {"n_prompt", toks.size()}, {"n_new", suffix.size()}, {"reused", common},
                      {"prompt_s", prompt_s}, {"prompt_tps", suffix.size() / std::max(prompt_s, 1e-9)},
                      {"counts", counts}, {"n_past", model->n_past()}});

                std::vector<int32_t> out;
                size_t shown = 0;
                std::string stop = "length";
                auto g0 = Clock::now();
                for (int i = 0; i < max_tokens; ++i) {
                    int32_t id = sample(lg, temperature, top_p, eng.rng);
                    if (id == tok.eos() || tok.is_control(id)) { stop = "end"; break; }
                    if (model->n_past() + 1 >= model->max_ctx()) { stop = "context"; break; }
                    out.push_back(id);
                    std::string text = tok.decode(out);
                    size_t ok = utf8_complete(text);
                    std::string delta = ok > shown ? text.substr(shown, ok - shown) : "";
                    shown = std::max(shown, ok);

                    lg = model->eval({id}, eng.threads, &routing);            // also tells us which experts it used
                    cached.push_back(id);
                    json picks = json::array();
                    for (int l = 0; l < L; ++l)
                        picks.push_back(std::vector<int32_t>(routing.begin() + (size_t)l * K, routing.begin() + (size_t)(l + 1) * K));
                    const double el = since(g0);
                    if (!send({{"type", "token"}, {"text", delta}, {"tps", (i + 1) / std::max(el, 1e-9)},
                               {"n_past", model->n_past()}, {"routing", picks}})) { stop = "abort"; break; }
                }
                const double gen_s = since(g0);
                send({{"type", "done"}, {"n_gen", out.size()}, {"gen_s", gen_s},
                      {"gen_tps", out.size() / std::max(gen_s, 1e-9)}, {"stop", stop}, {"n_past", model->n_past()}});
            } catch (const std::exception& e) {
                send({{"type", "error"}, {"message", e.what()}});
            }
            sink.done();
            return true;
        });
    });

    std::fprintf(stderr, "\n  tierlm is ready: open http://%s:%d in your browser\n  %s\n  (Ctrl+C to stop)\n\n",
                 a.host == "0.0.0.0" ? "127.0.0.1" : a.host.c_str(), a.port,
                 eng.model ? "" : "No model loaded yet: the page will show the setup screen.");
    if (!svr.listen(a.host, a.port)) {
        std::fprintf(stderr, "error: could not listen on %s:%d (is something else using that port? try --port 8041)\n", a.host.c_str(), a.port);
        return 1;
    }
    return 0;
}
