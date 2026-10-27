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
//   GET  /api/models   .gguf files in the models folder, and whether each is supported
//   POST /api/load     {"model": "file.gguf", "device": "auto|cpu|Vulkan0|...", "ctx": 4096, "threads": 4}
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

using namespace tierlm;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

namespace {

struct Args {
    std::string model, host = "127.0.0.1", device = "auto", web = "web", models = "models";
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


// The loaded model and everything that belongs to it.
struct Engine {
    std::unique_ptr<OlmoeModel> model;
    Tokenizer tok;
    std::vector<int32_t> cached;      // the tokens currently in the KV cache
    std::string path, name;
    int threads = 1;
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
        else if (k == "--models") a.models = next();
        else if (k == "-h" || k == "--help") {
            std::fprintf(stderr, "usage: %s [model.gguf] [--port 8080] [--host 127.0.0.1] [-d auto|cpu|gpu|<device>] [-t threads] [-c context] [--web dir] [--models dir]\n"
                                 "Without a model, the web page opens a setup screen to choose the hardware and model.\n", argv[0]);
            return 0;
        }
        else if (a.model.empty()) a.model = k;
    }

    Engine eng;
    eng.threads = a.threads;
    std::mutex mu;                       // one load or generation at a time

    auto load = [&](const std::string& path, const std::string& device, int ctx, int threads) {
        json info = describe_model(path);
        if (!info.value("supported", false)) throw std::runtime_error(info.value("reason", std::string("unsupported model")));
        eng.model.reset();               // free the old model (and its GPU memory) first
        eng.cached.clear();
        auto t0 = Clock::now();
        eng.tok = Tokenizer::from_gguf(path);
        eng.model = OlmoeModel::load(path, ctx, device);
        eng.path = path;
        eng.name = gguf_string(path, "general.name");
        eng.threads = threads > 0 ? threads : a.threads;
        eng.load_s = since(t0);
        std::fprintf(stderr, "loaded %s on %s in %.1fs\n", path.c_str(), eng.model->device_name().c_str(), eng.load_s);
    };
    if (!a.model.empty()) {
        try { load(a.model, a.device, a.ctx, a.threads); }
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

    json last_info = {{"loaded", false}};
    svr.Get("/api/info", [&](const httplib::Request&, httplib::Response& res) {
        // While a reply is streaming or a model is loading, answer from the last snapshot instead of waiting.
        std::unique_lock<std::mutex> lk(mu, std::try_to_lock);
        if (!lk.owns_lock()) { json j = last_info; j["busy"] = true; res.set_content(dump(j), "application/json"); return; }
        if (!eng.model) { last_info = {{"loaded", false}}; res.set_content(dump(last_info), "application/json"); return; }
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
        };
        last_info = j;
        res.set_content(dump(j), "application/json");
    });

    // Hardware the engine can use, with memory, for the setup screen.
    svr.Get("/api/devices", [&](const httplib::Request&, httplib::Response& res) {
        json list = json::array();
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
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d);
            list.push_back({{"id", ggml_backend_dev_name(d)}, {"description", ggml_backend_dev_description(d)},
                            {"type", type}, {"backend", reg ? ggml_backend_reg_name(reg) : ""},
                            {"free", free}, {"total", total}});
        }
        json j = {{"devices", list}, {"threads", a.threads}, {"hw_threads", (int)std::thread::hardware_concurrency()}};
        res.set_content(dump(j), "application/json");
    });

    // Models in the models folder, read from their headers (fast: no weights are loaded).
    svr.Get("/api/models", [&](const httplib::Request&, httplib::Response& res) {
        json list = json::array();
        std::error_code ec;
        for (const auto& f : std::filesystem::directory_iterator(std::filesystem::u8path(a.models), ec)) {
            if (!f.is_regular_file() || f.path().extension() != ".gguf") continue;
            list.push_back(describe_model(f.path().u8string()));
        }
        json j = {{"folder", std::filesystem::absolute(std::filesystem::u8path(a.models), ec).u8string()}, {"models", list},
                  {"default_ctx", a.ctx}, {"current", eng.path}};
        res.set_content(dump(j), "application/json");
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
            load(file.u8string(), body.value("device", std::string("auto")), body.value("ctx", a.ctx), body.value("threads", a.threads));
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
