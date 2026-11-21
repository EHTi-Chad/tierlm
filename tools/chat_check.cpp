// Test helper for the chat formats (tests/harmony_test.py): reads JSON from stdin and prints what tierlm-server would
// build or parse.
//   tierlm-chat-check prompt   [{"messages": [...], "tools": [...]}, ...]  -> each harmony prompt, followed by \x01
//   tierlm-chat-check reply    ["<raw reply>", ...]                        -> each {"display": ..., "calls": [...]}, one per line
//   tierlm-chat-check gemma4-prompt  [{"messages", "tools", "think": 0|1, "empty_thought": bool}, ...]  (tests/gemma4_chat_test.py)
//   tierlm-chat-check gemma4-reply   [{"raw": "...", "tools": [...]}, ...]
//   tierlm-chat-check inkling-prompt [{"messages", "tools", "think": 0|1}, ...]  (tests/inkling_chat_test.py)
//   tierlm-chat-check inkling-reply  ["<raw reply>", ...]
#include "harmony.h"
#include "gemma4.h"
#include "inkling.h"
#include <cstdio>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    std::stringstream ss;
    ss << std::cin.rdbuf();
    const auto in = nlohmann::ordered_json::parse(ss.str(), nullptr, false);
    if (in.is_discarded() || !in.is_array()) { std::fprintf(stderr, "expected a JSON array on stdin\n"); return 1; }
    if (mode == "prompt") {
        for (const auto& c : in)
            std::cout << harmony::prompt(nlohmann::json(c["messages"]), c.value("tools", nlohmann::ordered_json::array()),
                                         harmony::identity_of(c.value("template", std::string()))) << '\x01';
    } else if (mode == "reply") {
        for (const auto& r : in) {
            const auto p = harmony::parse(r.get<std::string>(), nlohmann::ordered_json::array());
            std::cout << nlohmann::json({{"display", p.content}, {"calls", p.calls}}).dump() << "\n";
        }
    } else if (mode == "gemma4-prompt") {
        for (const auto& c : in) {
            const auto tools = c.value("tools", nlohmann::ordered_json::array());
            std::cout << gemma4::prompt(agent::with_tools(nlohmann::json(c["messages"]), tools, agent::ToolFmt::Gemma4),
                                        c.value("think", 0), c.value("empty_thought", false), true) << '\x01';
        }
    } else if (mode == "gemma4-reply") {
        for (const auto& r : in) {
            const auto p = agent::parse_tool_calls(gemma4::display(r.value("raw", std::string())), r.value("tools", nlohmann::ordered_json::array()));
            std::cout << nlohmann::json({{"display", p.content}, {"calls", p.calls}}).dump() << "\n";
        }
    } else if (mode == "inkling-prompt") {
        for (const auto& c : in)
            std::cout << inkling::prompt(nlohmann::json(c["messages"]), c.value("tools", nlohmann::ordered_json::array()), c.value("think", 0)) << '\x01';
    } else if (mode == "inkling-reply") {
        for (const auto& r : in) {
            const auto p = inkling::parse(r.get<std::string>(), nlohmann::ordered_json::array());
            std::cout << nlohmann::json({{"display", p.content}, {"calls", p.calls}}).dump() << "\n";
        }
    } else { std::fprintf(stderr, "usage: tierlm-chat-check prompt|reply|gemma4-prompt|gemma4-reply|inkling-prompt|inkling-reply < input.json\n"); return 1; }
    return 0;
}
