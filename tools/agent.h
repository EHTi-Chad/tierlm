// Tool calling, for tierlm-server's OpenAI-compatible API (used by coding agents such as opencode, Cline and Cursor).
//
// A model is told about the tools in its system prompt, in the format it was trained on, and asks for one by writing
// a <tool_call> block in its reply. tierlm reads that block back out as an OpenAI-style tool call.
//   - Qwen3.5 / Qwen3.6 (and Qwen3-Coder): <tool_call><function=name><parameter=x>value</parameter></function></tool_call>
//   - Qwen3, Qwen2.5 and everything else ("Hermes" style): <tool_call>{"name": ..., "arguments": {...}}</tool_call>
//   - Gemma 4: <|tool_call>call:name{key:<|"|>text<|"|>,n:5}<tool_call|>, its tools declared in the system turn the
//     same way (<|tool>declaration:name{...}<tool|>), results back in the model's own turn
//     (<|tool_response>response:name{value:<|"|>...<|"|>}<tool_response|>), after which it carries on
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
using ojson = nlohmann::ordered_json;   // tools keep the order the app wrote them in (templates show them that way)

// ---------------------------------------------------------------- tool calling

enum class ToolFmt { QwenXML, Hermes, Gemma4 };

inline ToolFmt detect_tool_fmt(const std::string& chat_template) {
    if (chat_template.find("<|tool_call>call:") != std::string::npos) return ToolFmt::Gemma4;
    return chat_template.find("<function=") != std::string::npos ? ToolFmt::QwenXML : ToolFmt::Hermes;
}

// ---------------------------------------------------------------- Gemma 4's tool syntax
// Its own small object syntax, not JSON: {key:value,...} with bare keys, strings between <|"|> marks, true/false/null,
// numbers and [lists]. Written here as Gemma 4's chat template writes it (keys sorted, as its dictsort does).
namespace g4 {
inline const std::string Q = "<|\"|>";

inline std::string trim_ws(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}
inline std::string upper(std::string s) { for (char& c : s) c = (char)std::toupper((unsigned char)c); return s; }

// A JSON value as the template's format_argument writes it. escape_keys: object keys between <|"|> marks too.
template <class J>
inline std::string arg(const J& v, bool escape_keys = true) {
    if (v.is_null()) return "null";
    if (v.is_string()) return Q + v.template get<std::string>() + Q;
    if (v.is_boolean()) return v.template get<bool>() ? "true" : "false";
    if (v.is_object()) {
        std::vector<std::string> keys;
        for (auto it = v.begin(); it != v.end(); ++it) keys.push_back(it.key());
        std::sort(keys.begin(), keys.end());
        std::string s = "{";
        for (size_t i = 0; i < keys.size(); ++i)
            s += (i ? "," : "") + (escape_keys ? Q + keys[i] + Q : keys[i]) + ":" + arg(v[keys[i]], escape_keys);
        return s + "}";
    }
    if (v.is_array()) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + arg(v[i], escape_keys);
        return s + "]";
    }
    return v.dump();                                      // a number
}

template <class J>
inline std::string str(const J& o, const char* key) {   // o[key] as text ("" if missing or not text)
    return o.is_object() && o.contains(key) && o[key].is_string() ? o[key].template get<std::string>() : std::string();
}
template <class J>
inline std::string type_of(const J& v) {
    if (!v.is_object() || !v.contains("type")) return "";
    if (v["type"].is_string()) return upper(v["type"].template get<std::string>());
    if (v["type"].is_array() && !v["type"].empty() && v["type"][0].is_string()) return upper(v["type"][0].template get<std::string>());
    return "";
}
template <class J>
inline std::string quoted_list(const J& a) {
    std::string s = "[";
    if (a.is_array()) for (size_t i = 0; i < a.size(); ++i) s += (i ? "," : "") + Q + (a[i].is_string() ? a[i].template get<std::string>() : a[i].dump()) + Q;
    return s + "]";
}
template <class J>
inline bool truthy(const J& o, const char* key) {
    if (!o.is_object() || !o.contains(key)) return false;
    const J& v = o[key];
    return !(v.is_null() || (v.is_boolean() && !v.template get<bool>()) || (v.is_string() && v.template get<std::string>().empty()) ||
             ((v.is_array() || v.is_object()) && v.empty()));
}

// The template's format_parameters.
template <class J>
inline std::string params(const J& props, bool filter_keys = false) {
    static const char* standard[] = {"description", "type", "properties", "required", "nullable"};
    std::vector<std::string> keys;
    for (auto it = props.begin(); it != props.end(); ++it) keys.push_back(it.key());
    std::sort(keys.begin(), keys.end());
    std::string s;
    bool first = true;
    for (const auto& key : keys) {
        if (filter_keys && std::find(std::begin(standard), std::end(standard), key) != std::end(standard)) continue;
        const J& v = props[key];
        if (!first) s += ",";
        first = false;
        s += key + ":{";
        bool comma = false;
        auto sep = [&] { if (comma) s += ","; else comma = true; };
        if (truthy(v, "description")) { s += "description:" + Q + str(v, "description") + Q; comma = true; }
        const std::string t = type_of(v);
        if (t == "STRING") {
            if (truthy(v, "enum")) { sep(); s += "enum:" + arg(v["enum"]); }
        } else if (t == "ARRAY") {
            if (truthy(v, "items") && v["items"].is_object()) {
                sep();
                s += "items:{";
                const J& items = v["items"];
                std::vector<std::string> ik;
                for (auto it = items.begin(); it != items.end(); ++it) ik.push_back(it.key());
                std::sort(ik.begin(), ik.end());
                bool f2 = true;
                for (const auto& k : ik) {
                    const J& iv = items[k];
                    if (iv.is_null()) continue;
                    if (!f2) s += ",";
                    f2 = false;
                    if (k == "properties") s += "properties:{" + (iv.is_object() ? params(iv) : std::string()) + "}";
                    else if (k == "required") s += "required:" + quoted_list(iv);
                    else if (k == "type") {
                        if (iv.is_string()) s += "type:" + Q + upper(iv.template get<std::string>()) + Q;
                        else { J up = J::array(); if (iv.is_array()) for (const auto& x : iv) up.push_back(x.is_string() ? upper(x.template get<std::string>()) : x.dump()); s += "type:" + arg(up); }
                    } else s += k + ":" + arg(iv);
                }
                s += "}";
            }
        }
        if (truthy(v, "nullable")) { sep(); s += "nullable:true"; }
        if (t == "OBJECT") {
            sep();
            if (v.contains("properties") && v["properties"].is_object()) s += "properties:{" + params(v["properties"]) + "}";
            else s += "properties:{" + params(v, true) + "}";
            if (truthy(v, "required")) { sep(); s += "required:" + quoted_list(v["required"]); }
        }
        sep();
        s += "type:" + Q + t + Q + "}";
    }
    return s;
}

// One tool, as the template's format_function_declaration writes it, between <|tool> and <tool|>.
template <class J>
inline std::string declaration(const J& tool) {
    const J f = tool.is_object() && tool.contains("function") ? tool["function"] : tool;
    std::string s = "declaration:" + str(f, "name") + "{description:" + Q + str(f, "description") + Q;
    if (truthy(f, "parameters")) {
        const J& p = f["parameters"];
        s += ",parameters:{";
        if (truthy(p, "properties") && p["properties"].is_object()) s += "properties:{" + params(p["properties"]) + "},";
        if (truthy(p, "required")) s += "required:" + quoted_list(p["required"]) + ",";
        if (truthy(p, "type")) s += "type:" + Q + type_of(p) + Q + "}";
    }
    if (f.is_object() && f.contains("response")) {
        const J& r = f["response"];
        s += ",response:{";
        if (truthy(r, "description")) s += "description:" + Q + str(r, "description") + Q + ",";
        if (type_of(r) == "OBJECT") s += "type:" + Q + "OBJECT" + Q + "}";
    }
    s += "}";
    return trim_ws(s);
}

// Reads one value of the syntax at text[i], moving i past it. Unquoted text that isn't a JSON number, true, false or
// null is kept as text.
inline json parse_value(const std::string& t, size_t& i);
inline void skip_ws(const std::string& t, size_t& i) { while (i < t.size() && std::isspace((unsigned char)t[i])) ++i; }
inline json parse_value(const std::string& t, size_t& i) {
    skip_ws(t, i);
    if (t.compare(i, Q.size(), Q) == 0) {
        const size_t a = i + Q.size(), b = t.find(Q, a);
        i = b == std::string::npos ? t.size() : b + Q.size();
        return t.substr(a, (b == std::string::npos ? t.size() : b) - a);
    }
    if (i < t.size() && t[i] == '{') {
        json o = json::object();
        ++i;
        for (;;) {
            skip_ws(t, i);
            if (i >= t.size()) break;
            if (t[i] == '}') { ++i; break; }
            if (t[i] == ',') { ++i; continue; }
            std::string key;
            if (t.compare(i, Q.size(), Q) == 0) key = parse_value(t, i).get<std::string>();
            else { const size_t c = t.find(':', i); key = trim_ws(t.substr(i, c == std::string::npos ? std::string::npos : c - i)); i = c == std::string::npos ? t.size() : c; }
            skip_ws(t, i);
            if (i < t.size() && t[i] == ':') ++i;
            o[key] = parse_value(t, i);
        }
        return o;
    }
    if (i < t.size() && t[i] == '[') {
        json a = json::array();
        ++i;
        for (;;) {
            skip_ws(t, i);
            if (i >= t.size()) break;
            if (t[i] == ']') { ++i; break; }
            if (t[i] == ',') { ++i; continue; }
            a.push_back(parse_value(t, i));
        }
        return a;
    }
    size_t e = i;
    while (e < t.size() && t[e] != ',' && t[e] != '}' && t[e] != ']') ++e;
    const std::string w = trim_ws(t.substr(i, e - i));
    i = e;
    json j = json::parse(w, nullptr, false);
    return j.is_discarded() || j.is_object() || j.is_array() ? json(w) : j;
}
}  // namespace g4

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
template <class J>
inline J tool_function(const J& t) { return t.is_object() && t.contains("function") ? t["function"] : t; }

// The part of the system prompt that describes the tools.
inline std::string tools_prompt(ToolFmt f, const ojson& tools) {
    std::string s;
    if (f == ToolFmt::Gemma4) {
        for (const auto& t : tools) s += "<|tool>" + g4::declaration(t) + "<tool|>";
    } else if (f == ToolFmt::QwenXML) {
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
inline std::string system_with_tools(ToolFmt f, const std::string& system, const ojson& tools) {
    const std::string tp = tools_prompt(f, tools);
    if (system.empty()) return tp;
    if (f == ToolFmt::Gemma4) return g4::trim_ws(system) + tp;          // (straight after the system text, as its template)
    return f == ToolFmt::QwenXML ? tp + "\n\n" + system : system + "\n\n" + tp;
}

// Tool calls ([{"name", "arguments": {...}, "arguments_json": "..."}]) written the way the model writes them. The
// arguments are taken from arguments_json when there is one, which keeps them in the order the model wrote them
// (the parsed object is sorted), so the text matches the model's own and the cache can be reused., for an assistant message in the
// conversation that came from an app (tierlm's own page sends the reply exactly as the model wrote it).
inline std::string render_tool_calls(ToolFmt f, const json& calls, bool has_content) {
    std::string s;
    for (size_t i = 0; i < calls.size(); ++i) {
        json c = calls[i];
        if (c.contains("function") && c["function"].is_object()) c = c["function"];   // (OpenAI's shape, as sent)
        if (c.contains("arguments") && c["arguments"].is_string() && !c.contains("arguments_json")) c["arguments_json"] = c["arguments"];
        const std::string name = c.value("name", std::string());
        nlohmann::ordered_json args;                      // (null until found: arguments sent as an object count too)
        if (c.contains("arguments_json") && c["arguments_json"].is_string())
            args = nlohmann::ordered_json::parse(c["arguments_json"].get<std::string>(), nullptr, false);
        if (!args.is_object() && c.contains("arguments") && c["arguments"].is_object()) args = nlohmann::ordered_json(c["arguments"]);
        if (!args.is_object()) args = nlohmann::ordered_json::object();
        if (f == ToolFmt::Gemma4) {                       // (keys in the order the model wrote them: the cache matches)
            s += "<|tool_call>call:" + name + "{";
            bool first = true;
            for (auto it = args.begin(); it != args.end(); ++it) {
                s += (first ? "" : ",") + it.key() + ":" + g4::arg(it.value(), false);
                first = false;
            }
            s += "}<tool_call|>";
        } else if (f == ToolFmt::QwenXML) {
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
inline const char* kG4ToolOpen = "<|tool_call>";      // (Gemma 4: one token, never written in part)

// The first tool-call mark at or after `from`, in any of the formats (npos if none).
inline size_t find_tool_open(const std::string& text, size_t from = 0) {
    return std::min(text.find(kToolOpen, from), text.find(kG4ToolOpen, from));
}
inline bool has_tool_call(const std::string& text) { return find_tool_open(text) != std::string::npos; }

// Gemma 4: a tool's result, as its template writes it for an OpenAI-style message (the text as {value:...}, JSON or not).
inline std::string g4_tool_response(const std::string& name, const std::string& content) {
    return "<|tool_response>response:" + name + "{value:" + g4::arg(json(content), false) + "}<tool_response|>";
}

// Position of the first tool call in a reply (npos if none yet).
inline size_t tool_call_pos(const std::string& text) {
    const size_t from = answer_start(text);
    return from == std::string::npos ? std::string::npos : find_tool_open(text, from);
}

inline std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

// The JSON type a tool's parameter should have ("" if the tool or parameter is unknown).
inline std::string param_type(const ojson& tools, const std::string& tool, const std::string& param) {
    for (const auto& t : tools) {
        const ojson f = tool_function(t);
        if (!f.is_object() || f.value("name", std::string()) != tool) continue;
        const ojson p = f.value("parameters", ojson::object()).value("properties", ojson::object());
        if (p.contains(param) && p[param].is_object()) {
            const ojson& ty = p[param].contains("type") ? p[param]["type"] : ojson();
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
inline Parsed parse_tool_calls(const std::string& text, const ojson& tools) {
    Parsed r;
    size_t first = tool_call_pos(text);
    // A finished reply whose thinking was never closed (small models do that): a tool call in it is meant as one.
    if (first == std::string::npos && answer_start(text) == std::string::npos) first = find_tool_open(text, text.find("<think>"));
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
        if (text.compare(p, std::strlen(kG4ToolOpen), kG4ToolOpen) == 0) {   // Gemma 4: call:name{...}<tool_call|>
            const size_t body = p + std::strlen(kG4ToolOpen);
            const size_t end = text.find("<tool_call|>", body);
            std::string inner = text.substr(body, end == std::string::npos ? std::string::npos : end - body);
            inner = trim(inner);
            if (inner.compare(0, 5, "call:") == 0) inner.erase(0, 5);
            const size_t br = inner.find('{');
            const std::string name = trim(inner.substr(0, br));
            if (!name.empty()) {
                size_t i = br == std::string::npos ? inner.size() : br;
                json args = br == std::string::npos ? json::object() : g4::parse_value(inner, i);
                if (!args.is_object()) args = json::object();
                // Text the tool's description calls a number or a flag is turned into one (the model sometimes
                // quotes those); a string parameter stays text.
                nlohmann::ordered_json ordered = nlohmann::ordered_json::object();
                for (auto it = args.begin(); it != args.end(); ++it) {
                    const std::string ty = param_type(tools, name, it.key());
                    if (it.value().is_string() && !ty.empty() && ty != "string") {
                        json j = json::parse(trim(it.value().get<std::string>()), nullptr, false);
                        if (!j.is_discarded()) it.value() = j;
                    }
                }
                // (nlohmann's json sorts keys: the order the model wrote them is read again for arguments_json)
                {
                    size_t k = br == std::string::npos ? inner.size() : br + 1;
                    while (k < inner.size()) {
                        g4::skip_ws(inner, k);
                        if (k >= inner.size() || inner[k] == '}') break;
                        if (inner[k] == ',') { ++k; continue; }
                        std::string key;
                        if (inner.compare(k, g4::Q.size(), g4::Q) == 0) key = g4::parse_value(inner, k).get<std::string>();
                        else { const size_t c = inner.find(':', k); if (c == std::string::npos) break; key = trim(inner.substr(k, c - k)); k = c; }
                        g4::skip_ws(inner, k);
                        if (k < inner.size() && inner[k] == ':') ++k;
                        g4::parse_value(inner, k);
                        if (args.contains(key)) ordered[key] = nlohmann::ordered_json(args[key]);
                    }
                }
                r.calls.push_back({{"name", name}, {"arguments", args},
                                   {"arguments_json", ordered.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace)}});
            }
            if (end == std::string::npos) break;
            p = find_tool_open(text, end);
            continue;
        }
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
        p = find_tool_open(text, end);
    }
    return r;
}

// Tool calling in a conversation, made plain for the chat formats: the tools go into the system message, an
// assistant message's tool calls are written out the way the model writes them (unless its text already has them, as
// tierlm's page sends it), and tool results become a user turn of <tool_response> blocks, as Qwen's chat templates do.
// Gemma 4 instead has the results in the model's own turn, straight after its calls, then the message's own text (its
// template's order), and carries on writing there: such a message is marked "g4_open" (gemma4.h).
inline json with_tools(const json& messages, const ojson& tools, ToolFmt tf) {
    json out = json::array();
    const bool use = tools.is_array() && !tools.empty();
    size_t i = 0;
    std::string system;
    if (!messages.empty() && messages[0].value("role", std::string()) == "system") { system = messages[0].value("content", std::string()); i = 1; }
    if (use) out.push_back({{"role", "system"}, {"content", system_with_tools(tf, system, tools)}});
    else if (i == 1) out.push_back(messages[0]);
    for (; i < messages.size(); ++i) {
        const json& m = messages[i];
        const std::string role = m.value("role", std::string("user"));
        const bool calls = role == "assistant" && m.contains("tool_calls") && m["tool_calls"].is_array() && !m["tool_calls"].empty();
        if (tf == ToolFmt::Gemma4 && calls) {
            std::string text = m.value("content", std::string()), thought;
            const bool written = has_tool_call(text);       // (the page: the reply as the model wrote it)
            if (!written) {                                  // its thinking first (gemma4.h puts it in the thought channel)
                const size_t a = text.find("<think>"), b = a == std::string::npos ? a : text.find("</think>", a);
                if (b != std::string::npos) { thought = text.substr(a, b + 8 - a); text.erase(a, b + 8 - a); }
            }
            std::string c = written ? trim(text) : thought + render_tool_calls(tf, m["tool_calls"], false);
            std::vector<std::pair<std::string, std::string>> ids;   // (call id, tool name): which tool a result is from
            for (const auto& tc : m["tool_calls"]) {
                const json f = tc.contains("function") ? tc["function"] : tc;
                ids.push_back({tc.value("id", std::string()), f.value("name", std::string("unknown"))});
            }
            size_t k = 0;
            for (; i + 1 < messages.size() && messages[i + 1].value("role", std::string()) == "tool"; ++i, ++k) {
                const json& t = messages[i + 1];
                std::string name = t.value("name", std::string());
                for (const auto& [id, nm] : ids) if (!id.empty() && id == t.value("tool_call_id", std::string())) name = nm;
                if (name.empty()) name = k < ids.size() ? ids[k].second : "unknown";
                c += g4_tool_response(name, t.value("content", std::string()));
            }
            if (k == 0) c += "<|tool_response>";            // (no results yet: as its template leaves it)
            else if (!written) c += trim(text);
            out.push_back({{"role", "assistant"}, {"content", c}, {"g4_open", true}});
            continue;
        }
        if (role == "tool") {
            std::string c;
            for (; i < messages.size() && messages[i].value("role", std::string()) == "tool"; ++i)
                c += std::string(c.empty() ? "" : "\n") + "<tool_response>\n" + trim(messages[i].value("content", std::string())) + "\n</tool_response>";
            --i;
            out.push_back({{"role", "user"}, {"content", c}});
        } else if (calls) {
            std::string c = m.value("content", std::string());
            if (!has_tool_call(c)) c += render_tool_calls(tf, m["tool_calls"], !trim(c).empty());
            out.push_back({{"role", "assistant"}, {"content", c}});
        } else out.push_back({{"role", role}, {"content", m.value("content", std::string())}});
    }
    return out;
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
