// Gemma 4's chat format, for tierlm-server.
//
// Turns are <|turn>{system|user|model}\n...<turn|>\n. Thinking is switched on by <|think|> at the top of the system
// turn, and the model then thinks in a channel (<|channel>thought\n...<channel|>) before it answers. Tools are declared
// in the system turn (<|tool>declaration:...<tool|>, agent.h), the model calls one with <|tool_call>call:name{...}
// <tool_call|>, and the results go back inside the same model turn (<|tool_response>response:name{...}
// <tool_response|>), after which the model carries on: so a model turn can hold calls, results and the answer.
//
// The prompt follows Gemma 4's own chat template (in each file); tests/gemma4_chat_test.py checks it against the
// template run through Jinja. For the chat page and the API, the thought channel is shown as <think>...</think>.
#pragma once

#include "agent.h"

#include <string>

namespace gemma4 {

using json = nlohmann::json;

inline const std::string kThoughtOpen = "<|channel>thought\n";

// The reply as tierlm shows it: its thought channel as <think>...</think>. Grows only at the end as the reply does: a
// "thought" not yet whole after <|channel> is held back.
inline std::string display(const std::string& raw) {
    static const std::string open = "<|channel>", close = "<channel|>", label = "thought\n";
    std::string s;
    size_t i = 0;
    while (i < raw.size()) {
        const size_t a = raw.find(open, i);
        if (a == std::string::npos) { s += raw.substr(i); break; }
        s += raw.substr(i, a - i);
        size_t k = a + open.size();
        const size_t have = std::min(label.size(), raw.size() - k);
        if (raw.compare(k, have, label, 0, have) == 0) {
            if (have < label.size()) { s += "<think>"; break; }       // (the rest of "thought\n" is still to come)
            k += label.size();
        }
        const size_t b = raw.find(close, k);
        s += "<think>" + raw.substr(k, b == std::string::npos ? std::string::npos : b - k);
        if (b == std::string::npos) break;
        s += "</think>\n\n";
        i = b + close.size();
        while (i < raw.size() && raw[i] == '\n') ++i;                 // (the answer's own start)
    }
    return s;
}

// An earlier reply without its thinking, as the template's strip_thinking (tierlm keeps it as <think>...</think>;
// a raw thought channel is taken out too).
inline std::string strip_thinking(std::string c) {
    for (const auto& [o, e] : {std::pair<std::string, std::string>{"<think>", "</think>"}, {"<|channel>", "<channel|>"}})
        for (size_t a; (a = c.find(o)) != std::string::npos;) {
            const size_t b = c.find(e, a);
            c.erase(a, b == std::string::npos ? std::string::npos : b + e.size() - a);
        }
    return agent::trim(c);
}

// A reply in the current tool-call chain: its thinking (<think>...</think> as tierlm shows it) back in the thought
// channel, as the template writes reasoning_content there, then the rest.
inline std::string keep_thinking(const std::string& c) {
    const size_t a = c.find("<think>");
    if (a == std::string::npos) return strip_thinking(c);
    const size_t b = c.find("</think>", a);
    if (b == std::string::npos) return strip_thinking(c);
    const std::string thought = agent::trim(c.substr(a + 7, b - a - 7));
    const std::string rest = strip_thinking(c.substr(0, a) + c.substr(b + 8));
    return thought.empty() ? rest : "<|channel>thought\n" + thought + "\n<channel|>" + rest;
}

// The prompt. `messages` as agent::with_tools gives them (tools in the system message; an assistant message with
// tool calls has them and their results in its text, and "g4_open" when its turn carries on).
//   think          0 off, else on (<|think|> in the system turn)
//   empty_thought  thinking off: the reply starts after an empty thought channel (the 26B and 31B templates do that;
//                  the E2B and E4B ones don't, and those models start thinking aloud after one)
inline std::string prompt(const json& messages, int think, bool empty_thought, bool bos) {
    std::string s = bos ? "<bos>" : "";
    std::string sys;
    bool has_sys = false;
    for (const auto& m : messages)
        if (m.value("role", std::string()) == "system") {
            const std::string c = agent::trim(m.value("content", std::string()));
            sys += (has_sys && !c.empty() ? "\n\n" : "") + c;
            has_sys = true;
        }
    if (think != 0 || has_sys) s += "<|turn>system\n" + std::string(think != 0 ? "<|think|>\n" : "") + sys + "<turn|>\n";
    size_t last_user = 0;                                    // (replies after it keep their thinking: a tool-call chain)
    for (size_t i = 0; i < messages.size(); ++i) if (messages[i].value("role", std::string()) == "user") last_user = i + 1;
    bool open = false;                                       // in a model turn that carries on after tool results
    for (size_t i = 0; i < messages.size(); ++i) {
        const json& m = messages[i];
        const std::string role = m.value("role", std::string("user"));
        if (role == "system") continue;
        const bool model = role == "assistant";
        std::string c = m.value("content", std::string());
        if (model && i >= last_user) c = keep_thinking(c);
        else c = model ? strip_thinking(c) : agent::trim(c);
        if (open && !model) { s += "<turn|>\n"; open = false; }
        if (!open) s += "<|turn>" + std::string(model ? "model" : "user") + "\n";
        s += c;
        open = model && m.value("g4_open", false);
        if (!open) s += "<turn|>\n";
    }
    // The model carries on after its tools' results (thinking first, if on: the template opens the channel).
    if (open) {
        static const std::string res_end = "<tool_response|>";
        const bool after_results = s.size() >= res_end.size() && s.compare(s.size() - res_end.size(), res_end.size(), res_end) == 0;
        return think != 0 && after_results ? s + kThoughtOpen : s;
    }
    return s + "<|turn>model\n" + (think == 0 && empty_thought ? "<|channel>thought\n<channel|>" : "");
}

}  // namespace gemma4
