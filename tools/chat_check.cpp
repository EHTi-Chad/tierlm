// Test helper for the chat formats (tests/harmony_test.py): reads JSON from stdin and prints what tierlm-server would
// build or parse.
//   tierlm-chat-check prompt   [{"messages": [...], "tools": [...]}, ...]  -> each harmony prompt, followed by \x01
//   tierlm-chat-check reply    ["<raw reply>", ...]                        -> each {"display": ..., "calls": [...]}, one per line
#include "harmony.h"
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
    } else { std::fprintf(stderr, "usage: tierlm-chat-check prompt|reply < input.json\n"); return 1; }
    return 0;
}
