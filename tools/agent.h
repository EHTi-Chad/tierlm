// Tool calling, for tierlm-server's OpenAI-compatible API (used by coding agents such as opencode, Cline and Cursor).
//
// A model is told about the tools in its system prompt, in the format it was trained on, and asks for one by writing
// a <tool_call> block in its reply. tierlm reads that block back out as an OpenAI-style tool call.
//   - Qwen3.5 / Qwen3.6 (and Qwen3-Coder): <tool_call><function=name><parameter=x>value</parameter></function></tool_call>
//   - Qwen3, Qwen2.5 and everything else ("Hermes" style): <tool_call>{"name": ..., "arguments": {...}}</tool_call>
// The texts below follow the models' own chat templates, so the model sees what it saw in training. tierlm never
// runs a tool itself: the app that asked does, and sends the result back.
#pragma once

#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace agent {

using json = nlohmann::json;

// ---------------------------------------------------------------- tool calling

enum class ToolFmt { QwenXML, Hermes };

inline ToolFmt detect_tool_fmt(const std::string& chat_template) {
    return chat_template.find("<function=") != std::string::npos ? ToolFmt::QwenXML : ToolFmt::Hermes;
}

// JSON the way Python's json.dumps writes it (", " and ": "), as the chat templates' tojson does.
template <class J>
inline std::string pyjson(const J& j) {
    if (j.is_object()) {
        std::string s = "{";
        bool first = true;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (!first) s += ", ";
            first = false;
            s += J(it.key()).dump(-1, ' ', false, J::error_handler_t::replace) + ": " + pyjson(it.value());
        }
        return s + "}";
    }
    if (j.is_array()) {
        std::string s = "[";
        for (size_t i = 0; i < j.size(); ++i) s += (i ? ", " : "") + pyjson(j[i]);
        return s + "]";
    }
    return j.dump(-1, ' ', false, J::error_handler_t::replace);
}

// An OpenAI tool ({"type": "function", "function": {...}}) or a bare function description: the function part.
inline json tool_function(const json& t) { return t.is_object() && t.contains("function") ? t["function"] : t; }

// The part of the system prompt that describes the tools.
inline std::string tools_prompt(ToolFmt f, const json& tools) {
    std::string s;
    if (f == ToolFmt::QwenXML) {
        s = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto& t : tools) s += "\n" + pyjson(t);
        s += "\n</tools>";
        s += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
             "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
             "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
             "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
             "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
             "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
             "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
             "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
    } else {
        s = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (const auto& t : tools) s += "\n" + pyjson(t);
        s += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
             "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
    }
    return s;
}

// The system message with the tools added, where each model family expects them.
inline std::string system_with_tools(ToolFmt f, const std::string& system, const json& tools) {
    const std::string tp = tools_prompt(f, tools);
    if (system.empty()) return tp;
    return f == ToolFmt::QwenXML ? tp + "\n\n" + system : system + "\n\n" + tp;
}

// Tool calls ([{"name", "arguments": {...}, "arguments_json": "..."}]) written the way the model writes them. The
// arguments are taken from arguments_json when there is one, which keeps them in the order the model wrote them
// (the parsed object is sorted), so the text matches the model's own and the cache can be reused., for an assistant message in the
// conversation that came from an app (tierlm's own page sends the reply exactly as the model wrote it).
inline std::string render_tool_calls(ToolFmt f, const json& calls, bool has_content) {
    std::string s;
    for (size_t i = 0; i < calls.size(); ++i) {
        const json& c = calls[i];
        const std::string name = c.value("name", std::string());
        nlohmann::ordered_json args = nlohmann::ordered_json::object();
        if (c.contains("arguments_json") && c["arguments_json"].is_string())
            args = nlohmann::ordered_json::parse(c["arguments_json"].get<std::string>(), nullptr, false);
        if (!args.is_object() && c.contains("arguments") && c["arguments"].is_object()) args = nlohmann::ordered_json(c["arguments"]);
        if (!args.is_object()) args = nlohmann::ordered_json::object();
        if (f == ToolFmt::QwenXML) {
            s += i == 0 ? (has_content ? "\n\n" : "") : "\n";
            s += "<tool_call>\n<function=" + name + ">\n";
            for (auto it = args.begin(); it != args.end(); ++it)
                s += "<parameter=" + it.key() + ">\n" + (it.value().is_string() ? it.value().get<std::string>() : pyjson(it.value())) +
                     "\n</parameter>\n";
            s += "</function>\n</tool_call>";
        } else {
            if (i > 0 || has_content) s += "\n";
            s += "<tool_call>\n{\"name\": " + pyjson(nlohmann::ordered_json(name)) + ", \"arguments\": " + pyjson(args) + "}\n</tool_call>";
        }
    }
    return s;
}

// Where tool calls may start in a reply: after the thinking, if the model thinks. npos while it is still thinking.
inline size_t answer_start(const std::string& text) {
    const size_t open = text.find("<think>");
    if (open == std::string::npos) {
        const size_t close = text.find("</think>");      // a template that opened the thinking itself
        return close == std::string::npos ? 0 : close + 8;
    }
    const size_t close = text.find("</think>", open);
    return close == std::string::npos ? std::string::npos : close + 8;
}

inline const char* kToolOpen = "<tool_call>";

// Position of the first tool call in a reply (npos if none yet).
inline size_t tool_call_pos(const std::string& text) {
    const size_t from = answer_start(text);
    return from == std::string::npos ? std::string::npos : text.find(kToolOpen, from);
}

inline std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

// The JSON type a tool's parameter should have ("" if the tool or parameter is unknown).
inline std::string param_type(const json& tools, const std::string& tool, const std::string& param) {
    for (const auto& t : tools) {
        const json f = tool_function(t);
        if (!f.is_object() || f.value("name", std::string()) != tool) continue;
        const json p = f.value("parameters", json::object()).value("properties", json::object());
        if (p.contains(param) && p[param].is_object()) {
            const json& ty = p[param].contains("type") ? p[param]["type"] : json();
            if (ty.is_string()) return ty.get<std::string>();
            if (ty.is_array() && !ty.empty() && ty[0].is_string()) return ty[0].get<std::string>();
        }
    }
    return "";
}

struct Parsed {
    std::string content;          // the reply without its tool calls
    json calls = json::array();   // [{"name": ..., "arguments": {...}}]
};

// Reads the tool calls out of a finished reply.
inline Parsed parse_tool_calls(const std::string& text, const json& tools) {
    Parsed r;
    const size_t first = tool_call_pos(text);
    if (first == std::string::npos) {
        r.content = text;
        // Some models answer with a bare {"name": ..., "arguments": ...} object instead.
        const size_t from = answer_start(text);
        if (from != std::string::npos) {
            const std::string t = trim(text.substr(from));
            if (t.size() > 2 && t.front() == '{' && t.back() == '}') {
                json j = json::parse(t, nullptr, false);
                if (j.is_object() && j.contains("name") && j["name"].is_string()) {
                    bool known = false;
                    for (const auto& tl : tools) known |= tool_function(tl).value("name", std::string()) == j["name"].get<std::string>();
                    json args = j.contains("arguments") ? j["arguments"] : j.contains("parameters") ? j["parameters"] : json::object();
                    if (args.is_string()) args = json::parse(args.get<std::string>(), nullptr, false);
                    if (known && args.is_object()) {
                        r.calls.push_back({{"name", j["name"]}, {"arguments", args}});
                        r.content = text.substr(0, from);
                    }
                }
            }
        }
        return r;
    }
    std::string head = text.substr(0, first);
    while (!head.empty() && std::isspace((unsigned char)head.back())) head.pop_back();
    r.content = head;
    size_t p = first;
    while (p != std::string::npos) {
        const size_t body = p + std::strlen(kToolOpen);
        size_t end = text.find("</tool_call>", body);
        const std::string inner = text.substr(body, end == std::string::npos ? std::string::npos : end - body);
        const size_t fn = inner.find("<function=");
        if (fn != std::string::npos) {
            const size_t gt = inner.find('>', fn);
            if (gt != std::string::npos) {
                json call = {{"name", trim(inner.substr(fn + 10, gt - fn - 10))}, {"arguments", json::object()}};
                nlohmann::ordered_json ordered = nlohmann::ordered_json::object();
                const std::string name = call["name"];
                size_t q = gt;
                for (;;) {
                    const size_t ps = inner.find("<parameter=", q);
                    if (ps == std::string::npos) break;
                    const size_t pg = inner.find('>', ps);
                    if (pg == std::string::npos) break;
                    const std::string key = trim(inner.substr(ps + 11, pg - ps - 11));
                    size_t pe = inner.find("</parameter>", pg);
                    const size_t next = inner.find("<parameter=", pg);
                    if (pe == std::string::npos || (next != std::string::npos && next < pe)) pe = next;   // a missing close tag
                    if (pe == std::string::npos) { const size_t fe = inner.find("</function>", pg); pe = fe == std::string::npos ? inner.size() : fe; }
                    std::string v = inner.substr(pg + 1, pe - pg - 1);
                    if (!v.empty() && v.front() == '\n') v.erase(0, 1);
                    if (!v.empty() && v.back() == '\n') v.pop_back();
                    const std::string ty = param_type(tools, name, key);
                    json val = v;
                    if (ty != "string") {
                        json j = json::parse(trim(v), nullptr, false);
                        if (!j.is_discarded() && (ty.empty() ? !j.is_string() : true)) val = j;
                    }
                    call["arguments"][key] = val;
                    ordered[key] = nlohmann::ordered_json(val);
                    q = pe;
                }
                call["arguments_json"] = ordered.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace);
                r.calls.push_back(call);
            }
        } else {
            nlohmann::ordered_json j = nlohmann::ordered_json::parse(trim(inner), nullptr, false);
            if (j.is_object() && j.contains("name") && j["name"].is_string()) {
                nlohmann::ordered_json args = j.contains("arguments") ? j["arguments"] : j.contains("parameters") ? j["parameters"] : nlohmann::ordered_json::object();
                if (args.is_string()) { auto a2 = nlohmann::ordered_json::parse(args.get<std::string>(), nullptr, false); args = a2.is_discarded() ? nlohmann::ordered_json::object() : a2; }
                if (!args.is_object()) args = nlohmann::ordered_json::object();
                r.calls.push_back({{"name", j["name"].get<std::string>()}, {"arguments", json::parse(args.dump())},
                                   {"arguments_json", args.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace)}});
            }
        }
        if (end == std::string::npos) break;
        p = text.find(kToolOpen, end);
    }
    return r;
}

// A reply's thinking (<think>...</think>) apart from its answer, as OpenAI-style APIs send it ("reasoning_content"),
// so a coding agent shows it as thinking instead of as part of the answer. Fed the reply piece by piece while it
// streams; what comes out of each part never changes once sent.
struct ThinkSplit {
    std::string acc;
    size_t r_sent = 0, c_sent = 0;
    // Adds `delta` and returns the new reasoning and answer text. `final`: the reply is complete.
    std::pair<std::string, std::string> feed(const std::string& delta, bool final = false) {
        acc += delta;
        std::string R, C;
        size_t i = 0;
        while (i < acc.size() && std::isspace((unsigned char)acc[i])) ++i;
        static const std::string open = "<think>", close = "</think>";
        if (acc.compare(i, open.size(), open) == 0) {
            size_t rs = i + open.size();
            while (rs < acc.size() && acc[rs] == '\n') ++rs;
            const size_t ce = acc.find(close, rs);
            if (ce == std::string::npos) {
                R = acc.substr(rs);
                if (!final)                                    // hold back what could be the start of </think>
                    for (size_t k = std::min(close.size() - 1, R.size()); k > 0; --k)
                        if (R.compare(R.size() - k, k, close, 0, k) == 0) { R.resize(R.size() - k); break; }
                while (!R.empty() && R.back() == '\n') R.pop_back();   // sent once more text follows
            } else {
                R = acc.substr(rs, ce - rs);
                while (!R.empty() && R.back() == '\n') R.pop_back();
                size_t cs = ce + close.size();
                while (cs < acc.size() && std::isspace((unsigned char)acc[cs])) ++cs;
                C = acc.substr(cs);
            }
        } else if (!final && i == acc.size()) {
        } else if (!final && open.compare(0, acc.size() - i, acc, i, acc.size() - i) == 0) {
        } else C = acc.substr(i);
        std::pair<std::string, std::string> out;
        if (R.size() > r_sent) { out.first = R.substr(r_sent); r_sent = R.size(); }
        if (C.size() > c_sent) { out.second = C.substr(c_sent); c_sent = C.size(); }
        return out;
    }
};

// A finished reply split the same way: {reasoning, answer}.
inline std::pair<std::string, std::string> split_thinking(const std::string& text) {
    ThinkSplit t;
    return t.feed(text, true);
}

}  // namespace agent
