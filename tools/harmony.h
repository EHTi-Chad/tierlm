// OpenAI's "harmony" chat format (gpt-oss and models built on it), for tierlm-server.
//
// A conversation is a series of messages, each <|start|>{role}<|channel|>{channel}<|message|>{text}<|end|>. A reply
// is several messages in one go: the reasoning on the "analysis" channel, then the answer on the "final" channel,
// ending with <|return|>; or a tool call on the "commentary" channel addressed "to=functions.{name}", ending with
// <|call|>. So the reply stops only at <|return|> or <|call|>, not at the first special token.
//
// The prompt follows OpenAI's chat template (openai-gpt-oss-120b.jinja) exactly, tools rendered as a TypeScript
// namespace included; tests/harmony_test.py checks it against the template run through Jinja. For the chat page and
// the API, a reply is shown as "<think>{analysis}</think>\n\n{final}", like the other thinking models.
#pragma once

#include "agent.h"

#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

namespace harmony {

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;

// JSON as the template's tojson writes it (Python's json.dumps: ", " and ": ", non-ASCII kept).
inline std::string tojson(const nlohmann::ordered_json& j) { return agent::pyjson(j); }

// The model identity line of the file's own chat template ({%- set model_identity = "..." %}), or OpenAI's.
inline std::string identity_of(const std::string& tmpl) {
    const std::string key = "set model_identity = \"";
    const size_t a = tmpl.find(key);
    if (a != std::string::npos) {
        const size_t b = tmpl.find('"', a + key.size());
        if (b != std::string::npos) return tmpl.substr(a + key.size(), b - a - key.size());
    }
    return "You are ChatGPT, a large language model trained by OpenAI.";
}

template <class J>
inline bool has(const J& o, const char* k) { return o.is_object() && o.contains(k) && !o[k].is_null(); }

// render_typescript_type from the template
inline std::string ts_type(const ojson& spec) {
    if (!spec.is_object()) return "any";
    const ojson& ty = spec.contains("type") ? spec["type"] : ojson();
    const bool nullable = spec.value("nullable", false);
    if (ty.is_string() && ty == "array") {
        std::string s;
        if (has(spec, "items") && !spec["items"].empty()) {
            const ojson& it = spec["items"];
            const std::string t = it.is_object() && it.contains("type") && it["type"].is_string() ? it["type"].get<std::string>() : "";
            if (t == "string") s = "string[]";
            else if (t == "number" || t == "integer") s = "number[]";
            else if (t == "boolean") s = "boolean[]";
            else {
                const std::string inner = ts_type(it);
                s = inner == "object | object" || inner.size() > 50 ? "any[]" : inner + "[]";
            }
        } else s = "any[]";
        return nullable ? s + " | null" : s;
    }
    if (ty.is_array() && !ty.empty()) {
        if (ty.size() > 1) {
            std::string s;
            for (size_t i = 0; i < ty.size(); ++i) s += (i ? " | " : "") + (ty[i].is_string() ? ty[i].get<std::string>() : ty[i].dump());
            return s;
        }
        return ty[0].is_string() ? ty[0].get<std::string>() : ty[0].dump();
    }
    if (has(spec, "oneOf") && spec["oneOf"].is_array()) {
        bool obj = false;
        for (const auto& v : spec["oneOf"]) obj |= v.is_object() && v.value("type", std::string()) == "object";
        if (obj && spec["oneOf"].size() > 1) return "any";
        std::string s;
        for (size_t i = 0; i < spec["oneOf"].size(); ++i) {
            const ojson& v = spec["oneOf"][i];
            s += ts_type(v);
            if (has(v, "description")) s += "// " + v["description"].get<std::string>();
            if (v.is_object() && v.contains("default")) s += "\n// default: " + tojson(v["default"]);
            if (i + 1 < spec["oneOf"].size()) s += " | \n";
        }
        return s;
    }
    if (ty.is_string()) {
        const std::string t = ty;
        if (t == "string") {
            if (has(spec, "enum") && spec["enum"].is_array() && !spec["enum"].empty()) {
                std::string s = "\"";
                for (size_t i = 0; i < spec["enum"].size(); ++i)
                    s += (i ? "\" | \"" : "") + (spec["enum"][i].is_string() ? spec["enum"][i].get<std::string>() : spec["enum"][i].dump());
                return s + "\"";
            }
            return nullable ? "string | null" : "string";
        }
        if (t == "number" || t == "integer") return "number";
        if (t == "boolean") return "boolean";
        if (t == "object") {
            if (!has(spec, "properties") || spec["properties"].empty()) return "object";
            const ojson req = has(spec, "required") ? spec["required"] : ojson::array();
            std::string s = "{\n";
            size_t i = 0, n = spec["properties"].size();
            for (auto it = spec["properties"].begin(); it != spec["properties"].end(); ++it, ++i) {
                s += it.key();
                if (std::find(req.begin(), req.end(), it.key()) == req.end()) s += "?";
                s += ": \n                " + ts_type(it.value());   // (the template keeps this line break and indent)
                if (i + 1 < n) s += ", ";
            }
            return s + "}";
        }
    }
    return "any";
}

// render_tool_namespace("functions", tools)
inline std::string tools_namespace(const ojson& tools) {
    std::string s = "## functions\n\nnamespace functions {\n\n";
    for (const auto& t : tools) {
        const ojson f = agent::tool_function(t);
        if (!f.is_object()) continue;
        s += "// " + (has(f, "description") ? f["description"].get<std::string>() : std::string()) + "\n";
        s += "type " + f.value("name", std::string()) + " = ";
        const ojson p = has(f, "parameters") ? f["parameters"] : ojson::object();
        if (has(p, "properties") && !p["properties"].empty()) {
            const ojson req = has(p, "required") ? p["required"] : ojson::array();
            s += "(_: {\n";
            for (auto it = p["properties"].begin(); it != p["properties"].end(); ++it) {
                const ojson& ps = it.value();
                if (has(ps, "description")) s += "// " + ps["description"].get<std::string>() + "\n";
                s += it.key();
                if (std::find(req.begin(), req.end(), it.key()) == req.end()) s += "?";
                s += ": " + ts_type(ps);
                if (ps.is_object() && ps.contains("default")) {
                    const ojson& d = ps["default"];
                    const std::string raw = d.is_string() ? d.get<std::string>() : tojson(d);
                    if (has(ps, "enum")) s += ", // default: " + raw;
                    else if (has(ps, "oneOf")) s += "// default: " + raw;
                    else s += ", // default: " + tojson(d);
                }
                s += ",\n";
            }
            s += "}) => any;\n\n";
        } else s += "() => any;\n\n";
    }
    return s + "} // namespace functions";
}

inline std::string today() {
    if (const char* d = std::getenv("TIERLM_DATE")) return d;          // tests
    std::time_t t = std::time(nullptr);
    char buf[16];
    std::strftime(buf, sizeof buf, "%Y-%m-%d", std::localtime(&t));
    return buf;
}

// A message's reasoning and answer, from tierlm's "<think>...</think>answer" form (or reasoning_content).
inline void split_message(const json& m, std::string& thinking, std::string& content) {
    content = m.contains("content") && m["content"].is_string() ? m["content"].get<std::string>() : "";
    thinking = m.contains("thinking") && m["thinking"].is_string() ? m["thinking"].get<std::string>() : "";
    if (thinking.empty() && m.contains("reasoning_content") && m["reasoning_content"].is_string()) thinking = m["reasoning_content"];
    const size_t a = content.find("<think>");
    if (a != std::string::npos) {
        const size_t b = content.find("</think>", a);
        const std::string inner = content.substr(a + 7, b == std::string::npos ? std::string::npos : b - a - 7);
        if (thinking.empty()) thinking = agent::trim(inner);
        content = b == std::string::npos ? "" : agent::trim(content.substr(b + 8));
    }
}

// The prompt for a conversation (OpenAI's template with add_generation_prompt).
inline std::string prompt(const json& messages, const ojson& tools, const std::string& identity, const std::string& effort = "medium") {
    const bool use_tools = tools.is_array() && !tools.empty();
    std::string s = "<|start|>system<|message|>" + identity + "\nKnowledge cutoff: 2024-06\nCurrent date: " + today() +
                    "\n\nReasoning: " + effort + "\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.";
    if (use_tools) s += "\nCalls to these tools must go to the commentary channel: 'functions'.";
    s += "<|end|>";
    size_t first = 0;
    std::string dev;
    if (!messages.empty() && (messages[0].value("role", std::string()) == "system" || messages[0].value("role", std::string()) == "developer")) {
        dev = messages[0].value("content", std::string());
        first = 1;
    }
    if (!dev.empty() || use_tools) {
        s += "<|start|>developer<|message|>";
        if (!dev.empty()) s += "# Instructions\n\n" + dev + "\n\n";
        if (use_tools) s += "# Tools\n\n" + tools_namespace(tools);
        s += "<|end|>";
    }
    std::vector<std::string> pending;                 // names of the tool calls whose results come next
    std::string last_call;
    for (size_t i = first; i < messages.size(); ++i) {
        const json& m = messages[i];
        const std::string role = m.value("role", std::string("user"));
        if (role == "assistant") {
            std::string thinking, content;
            split_message(m, thinking, content);
            const bool calls = m.contains("tool_calls") && m["tool_calls"].is_array() && !m["tool_calls"].empty();
            if (calls) {
                // The reasoning before a tool call is kept, unless a later answer (an assistant message without
                // tool calls) follows: then, like all earlier reasoning, it's dropped.
                bool later_final = false;
                for (size_t j = i + 1; j < messages.size(); ++j)
                    later_final |= messages[j].value("role", std::string()) == "assistant" &&
                                   !(messages[j].contains("tool_calls") && messages[j]["tool_calls"].is_array() && !messages[j]["tool_calls"].empty());
                const std::string analysis = !content.empty() ? content : thinking;
                if (!analysis.empty() && !later_final) s += "<|start|>assistant<|channel|>analysis<|message|>" + analysis + "<|end|>";
                pending.clear();
                for (const auto& tc : m["tool_calls"]) {
                    const json f = agent::tool_function(tc);
                    const std::string name = f.value("name", std::string());
                    nlohmann::ordered_json args = nlohmann::ordered_json::object();
                    if (f.contains("arguments_json") && f["arguments_json"].is_string())
                        args = nlohmann::ordered_json::parse(f["arguments_json"].get<std::string>(), nullptr, false);
                    else if (f.contains("arguments") && f["arguments"].is_string())
                        args = nlohmann::ordered_json::parse(f["arguments"].get<std::string>(), nullptr, false);
                    else if (f.contains("arguments")) args = nlohmann::ordered_json(f["arguments"]);
                    if (args.is_discarded()) args = nlohmann::ordered_json::object();
                    s += "<|start|>assistant to=functions." + name + "<|channel|>commentary json<|message|>" + tojson(args) + "<|call|>";
                    pending.push_back(name);
                    last_call = name;
                }
            } else {
                s += "<|start|>assistant<|channel|>final<|message|>" + content + "<|end|>";
                last_call.clear();
                pending.clear();
            }
        } else if (role == "tool") {
            std::string name = last_call;
            if (!pending.empty()) { name = pending.front(); pending.erase(pending.begin()); }
            if (m.contains("name") && m["name"].is_string() && !m["name"].get<std::string>().empty()) name = m["name"];
            s += "<|start|>functions." + name + " to=assistant<|channel|>commentary<|message|>" +
                 tojson(nlohmann::ordered_json(m.value("content", std::string()))) + "<|end|>";
        } else if (role == "user") {
            s += "<|start|>user<|message|>" + m.value("content", std::string()) + "<|end|>";
        }
    }
    return s + "<|start|>assistant";
}

// One message of a reply: its header (role, channel, recipient) and text.
struct Message {
    std::string channel, recipient, text;
    bool closed = false;                              // ended by <|end|>, <|return|> or <|call|>
};

// The messages of a reply so far (the reply starts after the prompt's "<|start|>assistant"). A message whose header
// is still being written is left out.
inline std::vector<Message> messages_of(const std::string& raw) {
    std::vector<Message> out;
    static const std::string kMsg = "<|message|>", kChan = "<|channel|>";
    if (raw.find("<|") == std::string::npos) { if (!raw.empty()) out.push_back({"final", "", raw, false}); return out; }
    size_t p = 0;
    while (p < raw.size()) {
        const size_t m = raw.find(kMsg, p);
        if (m == std::string::npos) break;                        // header not finished yet
        const std::string head = raw.substr(p, m - p);
        Message msg;
        const size_t c = head.find(kChan);
        if (c != std::string::npos) {
            size_t e = c + kChan.size();
            while (e < head.size() && head[e] != ' ' && head.compare(e, 2, "<|") != 0) ++e;
            msg.channel = head.substr(c + kChan.size(), e - c - kChan.size());
        }
        const size_t to = head.find("to=");
        if (to != std::string::npos) {
            size_t e = to + 3;
            while (e < head.size() && head[e] != ' ' && head.compare(e, 2, "<|") != 0) ++e;
            msg.recipient = head.substr(to + 3, e - to - 3);
        }
        const size_t b = m + kMsg.size();
        size_t end = std::string::npos, elen = 0;
        for (const char* t : {"<|end|>", "<|return|>", "<|call|>"}) {
            const size_t q = raw.find(t, b);
            if (q < end) { end = q; elen = std::strlen(t); }
        }
        msg.text = raw.substr(b, end == std::string::npos ? std::string::npos : end - b);
        msg.closed = end != std::string::npos;
        out.push_back(msg);
        if (!msg.closed) break;
        p = end + elen;
    }
    return out;
}

// The reply as tierlm shows it: "<think>{analysis}</think>\n\n{answer}". Grows only at the end as the raw reply
// grows, so it can be streamed. Tool calls aren't shown (they're returned separately).
inline std::string display(const std::string& raw) {
    std::string s;
    for (const Message& m : messages_of(raw)) {
        if (m.recipient.rfind("functions.", 0) == 0) continue;
        if (m.channel == "analysis") { s += "<think>" + m.text; if (m.closed) s += "</think>\n\n"; }
        else { s += m.text; if (m.closed && m.channel == "commentary") s += "\n\n"; }
    }
    return s;
}

// The tool calls in a reply, like agent::parse_tool_calls.
inline agent::Parsed parse(const std::string& raw, const ojson& tools) {
    (void)tools;
    agent::Parsed r;
    r.content = display(raw);
    for (const Message& m : messages_of(raw)) {
        if (m.recipient.rfind("functions.", 0) != 0) continue;
        const std::string name = m.recipient.substr(10);
        const std::string text = agent::trim(m.text);
        nlohmann::ordered_json args = nlohmann::ordered_json::parse(text, nullptr, false);
        if (args.is_discarded() || !args.is_object()) args = nlohmann::ordered_json::object();
        r.calls.push_back({{"name", name}, {"arguments", json::parse(args.dump())},
                           {"arguments_json", args.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace)}});
    }
    return r;
}

}  // namespace harmony
