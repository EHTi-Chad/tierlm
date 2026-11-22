// Qwen3.5 / 3.6 / 3.8 (and Flash-Next) conversations as their chat templates write them, for tierlm-server.
//
// These templates share one layout (checked against the templates in the GGUF files, tests/qwen_chat_test.py):
//   - the system turn: [the reasoning-effort line, Qwen3.8 only, when thinking is on], the tools' descriptions,
//     then the system message(s), each part trimmed;
//   - every message's text trimmed of whitespace at both ends (Jinja's trim);
//   - an assistant turn is "<think>\n" + its reasoning + "\n</think>\n\n" + its text + its tool calls, the reasoning
//     empty when there is none. (Qwen3.8's template writes every earlier reply this way by default, 3.5 / 3.6 with
//     preserve_thinking; that is also exactly what the model itself wrote, so a hybrid model's cache, which can't
//     rewind, carries on through the last reply instead of reading it again.)
//   - tool results as one user turn of <tool_response> blocks;
//   - the reply starts "<think>\n" (thinking on) or "<think>\n\n</think>\n\n" (off).
// Qwen3.8's templates also take a reasoning effort: xhigh (the default; "high" counts as xhigh) and low each add a
// line of instructions at the top of the system turn, medium adds none.
#pragma once

#include "agent.h"

#include <string>
#include <vector>

namespace qwen {

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;

inline const char* const kXHigh = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
                                  "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
inline const char* const kLow = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
                                "without unnecessary elaboration.";

// A template of this family: ChatML with <function=...> tool calls and "<think>\n" + reasoning_content + "\n</think>\n\n"
// written before an assistant turn's text.
inline bool is_family(const std::string& ct) {
    return ct.find("<|im_start|>") != std::string::npos && ct.find("<function=") != std::string::npos &&
           ct.find("reasoning_content + '\\n</think>\\n\\n'") != std::string::npos;
}
// Qwen3.8: the template has the reasoning-effort lines (exactly these; another wording isn't guessed at)
inline bool has_effort(const std::string& ct) {
    return ct.find(kXHigh) != std::string::npos && ct.find(kLow) != std::string::npos;
}

// Python's str.strip(), which is what Jinja's trim does: ASCII and Unicode whitespace off both ends.
inline std::string strip(const std::string& s) {
    auto ws_at = [&](size_t i, size_t& len) -> bool {            // a whitespace character starting at i (UTF-8)
        const unsigned char c = (unsigned char)s[i];
        len = 1;
        if (c == ' ' || (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x1F)) return true;
        if (c == 0xC2 && i + 1 < s.size()) {                      // U+0085, U+00A0
            const unsigned char d = (unsigned char)s[i + 1];
            if (d == 0x85 || d == 0xA0) { len = 2; return true; }
        }
        if (i + 2 < s.size()) {
            const unsigned char d = (unsigned char)s[i + 1], e = (unsigned char)s[i + 2];
            if (c == 0xE1 && d == 0x9A && e == 0x80) { len = 3; return true; }                              // U+1680
            if (c == 0xE2 && d == 0x80 && ((e >= 0x80 && e <= 0x8A) || e == 0xA8 || e == 0xA9 || e == 0xAF)) { len = 3; return true; }   // U+2000-200A, 2028, 2029, 202F
            if (c == 0xE2 && d == 0x81 && e == 0x9F) { len = 3; return true; }                              // U+205F
            if (c == 0xE3 && d == 0x80 && e == 0x80) { len = 3; return true; }                              // U+3000
        }
        return false;
    };
    size_t a = 0, b = s.size(), len = 0;
    while (a < b && ws_at(a, len)) a += len;
    for (;;) {                                                    // from the end: the last character's start
        if (b <= a) break;
        size_t st = b - 1;
        while (st > a && ((unsigned char)s[st] & 0xC0) == 0x80) --st;
        if (ws_at(st, len) && st + len == b) b = st; else break;
    }
    return s.substr(a, b - a);
}

// An assistant message's reasoning and text: "reasoning_content" if sent, else what's inside <think>...</think> in
// the text (tierlm's own page sends the reply as the model wrote it). A <think> never closed is all reasoning.
inline void split_reply(const json& m, std::string& reasoning, std::string& text) {
    text = m.contains("content") && m["content"].is_string() ? m["content"].get<std::string>() : "";
    reasoning = m.contains("reasoning_content") && m["reasoning_content"].is_string() ? m["reasoning_content"].get<std::string>() : "";
    const size_t close = text.find("</think>");
    const size_t open = text.find("<think>");
    if (close != std::string::npos && (open == std::string::npos || open < close)) {
        const size_t from = open == std::string::npos ? 0 : open + 7;
        if (reasoning.empty()) reasoning = text.substr(from, close - from);
        text = text.substr(close + 8);
    } else if (open != std::string::npos && strip(text.substr(0, open)).empty()) {
        if (reasoning.empty()) reasoning = text.substr(open + 7);
        text.clear();
    }
    reasoning = strip(reasoning);
    text = strip(text);
}

// The prompt for a conversation (messages as tierlm-server normalizes them: role and text, an assistant's
// "tool_calls", images already marked in the text). think: 0 off, else on. effort: "low", "medium", anything else
// xhigh; used only when effort_lines (see has_effort).
inline std::string prompt(const json& messages, const ojson& tools, int think, const std::string& effort, bool effort_lines) {
    std::string s, sys;
    size_t i = 0;
    for (; i < messages.size(); ++i) {                            // the system messages at the start, joined
        const std::string role = messages[i].value("role", std::string("user"));
        if (role != "system" && role != "developer") break;
        const std::string c = strip(messages[i].value("content", std::string()));
        if (!c.empty()) sys += (sys.empty() ? "" : "\n") + c;
    }
    std::string instr;
    if (effort_lines && think != 0) instr = effort == "low" ? kLow : effort == "medium" ? "" : kXHigh;
    const bool use_tools = tools.is_array() && !tools.empty();
    if (use_tools) {
        s += "<|im_start|>system\n" + (instr.empty() ? "" : instr + "\n\n") + agent::tools_prompt(agent::ToolFmt::QwenXML, tools) +
             (sys.empty() ? "" : "\n\n" + sys) + "<|im_end|>\n";
    } else if (!sys.empty() || !instr.empty()) {
        s += "<|im_start|>system\n" + instr + (!instr.empty() && !sys.empty() ? "\n\n" : "") + sys + "<|im_end|>\n";
    }
    for (; i < messages.size(); ++i) {
        const json& m = messages[i];
        const std::string role = m.value("role", std::string("user"));
        if (role == "assistant") {
            std::string reasoning, text;
            split_reply(m, reasoning, text);
            s += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + text;
            if (m.contains("tool_calls") && m["tool_calls"].is_array() && !m["tool_calls"].empty() && !agent::has_tool_call(text))
                s += agent::render_tool_calls(agent::ToolFmt::QwenXML, m["tool_calls"], !text.empty());
            s += "<|im_end|>\n";
        } else if (role == "tool") {
            s += "<|im_start|>user";
            for (; i < messages.size() && messages[i].value("role", std::string()) == "tool"; ++i)
                s += "\n<tool_response>\n" + strip(messages[i].value("content", std::string())) + "\n</tool_response>";
            --i;
            s += "<|im_end|>\n";
        } else {                                                  // user (a system message later on is written as it comes)
            s += "<|im_start|>" + (role == "system" || role == "developer" ? std::string("system") : std::string("user")) + "\n" +
                 strip(m.value("content", std::string())) + "<|im_end|>\n";
        }
    }
    return s + "<|im_start|>assistant\n" + (think == 0 ? "<think>\n\n</think>\n\n" : "<think>\n");
}

}  // namespace qwen
