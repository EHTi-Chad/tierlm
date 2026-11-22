// GLM's chat format (Z.ai: GLM-4.5, 4.6, 4.7 and 4.7-Flash), for tierlm-server.
//
// "[gMASK]<sop>", then the turns, each opened by a special token and closed by the next: <|system|>, <|user|>,
// <|assistant|> and <|observation|> (tool results, each in <tool_response>...</tool_response>). The tools are a system
// turn of their own at the start, one JSON line each. An earlier reply's thinking is written only after the last user
// message (before it, an empty thinking). A tool call is <tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE
// </arg_value>...</tool_call>, a value as text when it is text and as JSON otherwise.
//
// Two layouts, told apart by the model's own template:
//   v46 (GLM-4.5 and 4.6): a newline after each turn's token, "<think></think>" for an earlier reply's thinking,
//       the tool call's parts on lines of their own; thinking off is "/nothink" at the end of each user message and an
//       empty "<think></think>" after the reply's turn token; thinking on leaves the model to open its own.
//   v47 (GLM-4.7-Flash): no newlines; an earlier reply's empty thinking is a bare "</think>"; the reply's turn ends with
//       "<think>" (thinking on) or "</think>" (off).
// As the templates llama.cpp keeps (models/templates/GLM-4.6.jinja and GLM-4.7-Flash.jinja), checked character for
// character by tests/glm_chat_test.py. The reply's tool calls are read by agent::parse_tool_calls.
#pragma once

#include "agent.h"

#include <string>

namespace glm {

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;

enum class Layout { v46, v47 };

// Which layout a template writes (v47 unless it has the earlier one's "/nothink").
inline Layout layout_of(const std::string& chat_template) {
    return chat_template.find("/nothink") != std::string::npos ? Layout::v46 : Layout::v47;
}

inline bool is_glm(const std::string& chat_template) {
    return chat_template.find("[gMASK]<sop>") != std::string::npos && chat_template.find("<|observation|>") != std::string::npos;
}

// Python's str.strip(), lstrip('\n') and rstrip('\n')
inline std::string strip(const std::string& s) { return agent::trim(s); }
inline std::string lstrip_nl(std::string s) { size_t a = 0; while (a < s.size() && s[a] == '\n') ++a; return s.substr(a); }
inline std::string rstrip_nl(std::string s) { while (!s.empty() && s.back() == '\n') s.pop_back(); return s; }

// A message's text: a string, or the text parts of a list
inline std::string visible_text(const json& m) {
    if (!m.contains("content")) return "";
    const json& c = m["content"];
    if (c.is_string()) return c.get<std::string>();
    if (c.is_array()) {
        std::string s;
        for (const auto& it : c) {
            if (it.is_object() && it.value("type", std::string()) == "text" && it.contains("text") && it["text"].is_string()) s += it["text"].get<std::string>();
            else if (it.is_string()) s += it.get<std::string>();
        }
        return s;
    }
    return c.is_null() ? "" : c.dump();
}

// The prompt for a conversation, the reply's turn opened. think: 0 off, else on.
inline std::string prompt(const json& messages, const ojson& tools, int think, Layout lay) {
    const bool v46 = lay == Layout::v46;
    const std::string nl = v46 ? "\n" : "";
    std::string s = "[gMASK]<sop>";
    if (tools.is_array() && !tools.empty()) {
        s += "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
             "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
        for (const auto& t : tools) s += agent::pyjson(t) + "\n";
        s += "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n";
        s += v46 ? "<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n"
                   "<arg_key>{arg-key-2}</arg_key>\n<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call>"
                 : "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
                   "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";
    }
    int last_user = -1;
    for (size_t i = 0; i < messages.size(); ++i) if (messages[i].value("role", std::string()) == "user") last_user = (int)i;
    for (size_t i = 0; i < messages.size(); ++i) {
        const json& m = messages[i];
        const std::string role = m.value("role", std::string("user"));
        if (role == "user") {
            const std::string t = visible_text(m);
            s += "<|user|>" + nl + t;
            const std::string nt = "/nothink";
            if (v46 && think == 0 && !(t.size() >= nt.size() && t.compare(t.size() - nt.size(), nt.size(), nt) == 0)) s += nt;
        } else if (role == "assistant") {
            s += "<|assistant|>";
            std::string reasoning, content = visible_text(m);
            if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) reasoning = m["reasoning_content"].get<std::string>();
            else if (content.find("</think>") != std::string::npos) {
                std::string before = rstrip_nl(content.substr(0, content.find("</think>")));
                const size_t o = before.rfind("<think>");
                reasoning = lstrip_nl(o == std::string::npos ? before : before.substr(o + 7));
                content = lstrip_nl(content.substr(content.rfind("</think>") + 8));
            }
            if ((int)i > last_user && !reasoning.empty()) s += nl + "<think>" + strip(reasoning) + "</think>";
            else s += v46 ? "\n<think></think>" : "</think>";
            if (!strip(content).empty()) s += nl + strip(content);
            if (m.contains("tool_calls") && m["tool_calls"].is_array() && !m["tool_calls"].empty()) {
                for (const auto& tc : m["tool_calls"]) {
                    const json f = agent::tool_function(tc);
                    ojson args = ojson::object();
                    if (f.contains("arguments_json") && f["arguments_json"].is_string())
                        args = ojson::parse(f["arguments_json"].get<std::string>(), nullptr, false);
                    else if (f.contains("arguments") && f["arguments"].is_string())
                        args = ojson::parse(f["arguments"].get<std::string>(), nullptr, false);
                    else if (f.contains("arguments")) args = ojson::parse(f["arguments"].dump(), nullptr, false);
                    if (args.is_discarded() || !args.is_object()) args = ojson::object();
                    s += nl + "<tool_call>" + f.value("name", std::string()) + nl;
                    for (auto it = args.begin(); it != args.end(); ++it)
                        s += "<arg_key>" + it.key() + "</arg_key>" + nl + "<arg_value>" +
                             (it.value().is_string() ? it.value().get<std::string>() : agent::pyjson(it.value())) + "</arg_value>" + nl;
                    s += "</tool_call>";
                }
            }
        } else if (role == "tool") {
            if (i == 0 || messages[i - 1].value("role", std::string()) != "tool") s += "<|observation|>";
            const std::string c = m.contains("content") && m["content"].is_string() ? m["content"].get<std::string>()
                                : m.contains("content") && !m["content"].is_null() ? m["content"].dump() : "";
            s += v46 ? "\n<tool_response>\n" + c + "\n</tool_response>" : "<tool_response>" + c + "</tool_response>";
        } else if (role == "system" || role == "developer") {
            s += "<|system|>" + nl + visible_text(m);
        }
    }
    s += "<|assistant|>";
    if (v46) { if (think == 0) s += "\n<think></think>"; }
    else s += think == 0 ? "</think>" : "<think>";
    return s;
}

// What the reply begins with in tierlm's "<think>...</think>answer" form, for what the prompt opened: v47 with thinking
// on opened the thinking itself.
inline std::string reply_prefix(int think, Layout lay) { return lay == Layout::v47 && think != 0 ? "<think>" : ""; }

}  // namespace glm
