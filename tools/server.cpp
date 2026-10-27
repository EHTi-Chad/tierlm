// tierlm-server: runs the engine behind a local web page with a live dashboard.
//
//   tierlm-server model.gguf [--port 8080] [--host 127.0.0.1] [-d auto|cpu|gpu] [-t threads] [-c context] [--web dir]
//
// Endpoints
//   GET  /            the web page (web/index.html, re-read on every load so it can be edited live)
//   GET  /api/info    model, device and memory details
//   POST /api/chat    {"messages":[{"role":"user","content":"..."}], "temperature":0.7, "top_p":0.9, "max_tokens":512}
//                     streams Server-Sent Events: one "prompt" event, a "token" event per token
//                     (with the experts picked in every layer), then "done"
//   POST /api/reset   clears the KV cache
#include "olmoe.h"
#include "tokenizer.h"
#include "gguf.h"
#include "httplib.h"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    std::string model, host = "127.0.0.1", device = "auto", web = "web";
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
        else if (a.model.empty()) a.model = k;
    }
    if (a.model.empty()) {
        std::fprintf(stderr, "usage: %s model.gguf [--port 8080] [--host 127.0.0.1] [-d auto|cpu|gpu] [-t threads] [-c context] [--web dir]\n", argv[0]);
        return 1;
    }

    std::unique_ptr<OlmoeModel> model;
    Tokenizer tok;
    try {
        auto t0 = Clock::now();
        tok = Tokenizer::from_gguf(a.model);
        model = OlmoeModel::load(a.model, a.ctx, a.device);
        std::fprintf(stderr, "device: %s | loaded in %.1fs\n", model->device_name().c_str(), since(t0));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    const std::string name = gguf_string(a.model, "general.name");
    const auto& hp = model->hparams();

    std::mutex mu;                       // one generation at a time
    std::vector<int32_t> cached;         // the tokens currently in the KV cache
    std::mt19937 rng(std::random_device{}());

    httplib::Server svr;

    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        std::string page = read_file(a.web + "/index.html");
        if (page.empty()) {
            res.status = 404;
            res.set_content("web/index.html not found. Run the server from the project folder, or pass --web <folder>.", "text/plain");
            return;
        }
        res.set_content(page, "text/html; charset=utf-8");
    });

    svr.Get("/api/info", [&](const httplib::Request&, httplib::Response& res) {
        size_t free = 0, total = 0;
        model->device_memory(free, total);
        json j = {
            {"name", name.empty() ? "OLMoE" : name}, {"arch", "olmoe"},
            {"layers", hp.n_layer}, {"experts", hp.n_expert}, {"experts_used", hp.n_expert_used},
            {"vocab", hp.n_vocab}, {"ctx", model->max_ctx()}, {"n_past", model->n_past()},
            {"device", model->device_name()}, {"on_gpu", model->on_gpu()}, {"threads", a.threads},
            {"weights_bytes", model->weights_bytes()}, {"kv_bytes", model->kv_bytes()},
            {"device_free", free}, {"device_total", total},
        };
        res.set_content(dump(j), "application/json");
    });

    svr.Post("/api/reset", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(mu);
        model->reset();
        cached.clear();
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
            res.set_content("{\"error\":\"busy: a reply is already being generated\"}", "application/json");
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
                std::vector<float> lg = model->eval(suffix, a.threads, &routing);
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
                    int32_t id = sample(lg, temperature, top_p, rng);
                    if (id == tok.eos() || tok.is_control(id)) { stop = "end"; break; }
                    if (model->n_past() + 1 >= model->max_ctx()) { stop = "context"; break; }
                    out.push_back(id);
                    std::string text = tok.decode(out);
                    size_t ok = utf8_complete(text);
                    std::string delta = ok > shown ? text.substr(shown, ok - shown) : "";
                    shown = std::max(shown, ok);

                    lg = model->eval({id}, a.threads, &routing);            // also tells us which experts it used
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

    std::fprintf(stderr, "\n  tierlm is ready: open http://%s:%d in your browser\n  (Ctrl+C to stop)\n\n",
                 a.host == "0.0.0.0" ? "localhost" : a.host.c_str(), a.port);
    if (!svr.listen(a.host, a.port)) {
        std::fprintf(stderr, "error: could not listen on %s:%d (is something else using that port? try --port 8081)\n", a.host.c_str(), a.port);
        return 1;
    }
    return 0;
}
