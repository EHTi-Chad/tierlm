// Inkling 2's chat format (Thinking Machines), for tierlm-server.
//
// A conversation is a list of blocks, each "<|message_ROLE|>[name]<|content_KIND|>...<|end_message|>": roles user,
// model (the reply), system and tool; kinds text, thinking, invoke_tool_json (a tool call) and xml (the tool list).
// A system block "Thinking effort level: E" comes before the first message that isn't a system one (0: no
// thinking). The model's reply is one or more blocks, <|message_model|><|content_thinking|>...<|end_message|>, then
// <|message_model|><|content_text|>...<|end_message|> or tool calls <|message_model|>NAME<|content_invoke_tool_json|>
// {"name":NAME,"args":{...}}<|end_message|>, and ends with <|content_model_end_sampling|>. There is no generation
// prompt: the model writes its own first <|message_model|>.
//
// As Thinking Machines' chat template (llama.cpp keeps a copy: models/templates/thinkingmachines-Inkling-2-Small.jinja)
// and llama.cpp's parser of the reply (common/parsers/inkling.cpp). Shown as <think>thinking</think>answer.
#pragma once

#include "agent.h"

#include <string>
#include <vector>

namespace inkling {

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;

inline const std::string kModel = "<|message_model|>", kEnd = "<|end_message|>", kText = "<|content_text|>",
                         kThink = "<|content_thinking|>", kTool = "<|content_invoke_tool_json|>",
                         kStop = "<|content_model_end_sampling|>";

// The thinking effort's block: on, Thinking Machines' default (0.9); off, 0
inline std::string effort_block(int think) {
    return std::string("<|message_system|><|content_text|>Thinking effort level: ") + (think == 0 ? "0" : "0.9") + kEnd;
}

// A message's reasoning and answer, from tierlm's "<think>...</think>answer" form (or reasoning_content)
inline void split_message(const json& m, std::string& thinking, std::string& content) {
    content = m.contains("content") && m["content"].is_string() ? m["content"].get<std::string>() : "";
    thinking = m.contains("reasoning_content") && m["reasoning_content"].is_string() ? m["reasoning_content"].get<std::string>() : "";
    const size_t a = content.find("<think>");
    if (a != std::string::npos) {
        const size_t b = content.find("</think>", a);
        const std::string inner = content.substr(a + 7, b == std::string::npos ? std::string::npos : b - a - 7);
        if (thinking.empty()) thinking = agent::trim(inner);
        content = b == std::string::npos ? "" : agent::trim(content.substr(b + 8));
    }
}

// JSON as Jinja's tojson writes it here: no spaces, keys sorted (the tool list) or as given (a call's values)
inline std::string compact(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }
inline std::string compact(const ojson& j) { return j.dump(-1, ' ', false, ojson::error_handler_t::replace); }

// The prompt for a conversation. think: 0 off, else on.
inline std::string prompt(const json& messages, const ojson& tools, int think) {
    std::string s;
    if (tools.is_array() && !tools.empty()) {
        json specs = json::array();
        for (const auto& t : tools) {
            const json tj = json::parse(t.dump());
            const json f = tj.contains("function") ? tj["function"] : tj;
            specs.push_back({{"description", f.contains("description") && f["description"].is_string() ? f["description"] : json("")},
                             {"name", f.value("name", std::string())},
                             {"parameters", f.contains("parameters") && f["parameters"].is_object() ? f["parameters"] : json::object()},
                             {"type", tj.value("type", std::string("function"))}});
        }
        s += "<|message_system|>tool_declare<|content_xml|>" + compact(specs) + kEnd;
    }
    bool effort = false;
    std::vector<std::string> pending;                    // the tool calls whose results come next (for their names)
    for (const auto& m : messages) {
        const std::string role = m.value("role", std::string("user"));
        if (!effort && role != "system") { s += effort_block(think); effort = true; }
        if (role == "system" || role == "user" || role == "developer") {
            s += (role == "user" ? "<|message_user|>" : "<|message_system|>") + kText + m.value("content", std::string()) + kEnd;
        } else if (role == "tool") {
            std::string name = m.contains("name") && m["name"].is_string() ? m["name"].get<std::string>() : "";
            const std::string id = m.contains("tool_call_id") && m["tool_call_id"].is_string() ? m["tool_call_id"].get<std::string>() : "";
            if (name.empty() && !id.empty())                     // (as the template: the call with that id)
                for (const auto& p : messages)
                    if (p.value("role", std::string()) == "assistant" && p.contains("tool_calls") && p["tool_calls"].is_array())
                        for (const auto& tc : p["tool_calls"])
                            if (tc.is_object() && tc.value("id", std::string()) == id) name = agent::tool_function(tc).value("name", name);
            if (name.empty() && !pending.empty()) name = pending.front();
            if (!pending.empty()) pending.erase(pending.begin());
            const std::string c = m.contains("content") && m["content"].is_string() ? m["content"].get<std::string>()
                                : m.contains("content") ? compact(m["content"]) : "";
            s += "<|message_tool|>" + name + kText + c + kEnd;
        } else if (role == "assistant") {
            std::string thinking, content;
            split_message(m, thinking, content);
            const bool calls = m.contains("tool_calls") && m["tool_calls"].is_array() && !m["tool_calls"].empty();
            if (!thinking.empty()) s += kModel + kThink + thinking + kEnd;
            if (!content.empty() || !(calls || !thinking.empty())) s += kModel + kText + content + kEnd;
            pending.clear();
            if (calls)
                for (const auto& tc : m["tool_calls"]) {
                    const json f = agent::tool_function(tc);
                    const std::string name = f.value("name", std::string());
                    ojson args = ojson::object();
                    if (f.contains("arguments_json") && f["arguments_json"].is_string())
                        args = ojson::parse(f["arguments_json"].get<std::string>(), nullptr, false);
                    else if (f.contains("arguments") && f["arguments"].is_string())
                        args = ojson::parse(f["arguments"].get<std::string>(), nullptr, false);
                    else if (f.contains("arguments")) args = ojson::parse(f["arguments"].dump(), nullptr, false);
                    if (args.is_discarded() || !args.is_object()) args = ojson::object();
                    std::vector<std::string> keys;                // (dictsort, case-sensitive)
                    for (auto it = args.begin(); it != args.end(); ++it) keys.push_back(it.key());
                    std::sort(keys.begin(), keys.end());
                    std::string a = "{";
                    for (size_t k = 0; k < keys.size(); ++k) a += (k ? "," : "") + compact(json(keys[k])) + ":" + compact(args[keys[k]]);
                    s += kModel + name + kTool + "{\"name\":" + compact(json(name)) + ",\"args\":" + a + "}}" + kEnd;
                    pending.push_back(name);
                }
            s += kStop;
        }
    }
    if (!effort) s += effort_block(think);
    return s;
}

// One block of a reply: its kind ("text", "thinking", "tool" or "" while its header is still coming), the tool's
// name, its text, and whether its <|end_message|> has come.
struct Block { std::string kind, name, text; bool closed = false; };

inline std::vector<Block> blocks_of(const std::string& raw) {
    std::vector<Block> out;
    size_t p = 0;
    if (raw.compare(0, kModel.size(), kModel) != 0 && raw.find("<|") == std::string::npos) {   // (no blocks: plain text)
        if (!raw.empty()) out.push_back({"text", "", raw, false});
        return out;
    }
    while (p < raw.size()) {
        const size_t h = raw.find(kModel, p);
        if (h == std::string::npos) break;
        size_t b = h + kModel.size();
        Block blk;
        const size_t lt = raw.find("<|", b);
        // Text straight after <|message_model|>, with no content tag (Mini did that with its thinking tag not allowed:
        // 4,771 tokens that showed nothing): a tool's name is one short word, so anything longer, or with a space or
        // a line break in it, is the reply's text.
        auto prose = [](const std::string& h) { return h.size() > 64 || h.find_first_of(" \n\t") != std::string::npos; };
        if (lt == std::string::npos) {                                      // (the header isn't finished, or is text)
            if (prose(raw.substr(b))) out.push_back({"text", "", raw.substr(b), false});
            break;
        }
        const std::string head = raw.substr(b, lt - b), tag = raw.substr(lt);
        if (!head.empty() && (prose(head) || tag.compare(0, kEnd.size(), kEnd) == 0) && tag.compare(0, kTool.size(), kTool) != 0) {
            const size_t e = raw.find(kEnd, b);                              // (text up to its end, whatever tags it holds)
            out.push_back({"text", "", raw.substr(b, e == std::string::npos ? std::string::npos : e - b), e != std::string::npos});
            if (e == std::string::npos) break;
            p = e + kEnd.size();
            continue;
        }
        auto starts = [&](const std::string& t) { return tag.compare(0, t.size(), t) == 0; };
        if (starts(kText)) { blk.kind = "text"; b = lt + kText.size(); }
        else if (starts(kThink)) { blk.kind = "thinking"; b = lt + kThink.size(); }
        else if (starts(kTool)) { blk.kind = "tool"; blk.name = head; b = lt + kTool.size(); }
        else if (tag.find("|>") == std::string::npos) break;                // (the kind isn't finished)
        else { blk.kind = "other"; b = raw.find("|>", lt) + 2; }
        const size_t e = raw.find(kEnd, b);
        blk.text = raw.substr(b, e == std::string::npos ? std::string::npos : e - b);
        blk.closed = e != std::string::npos;
        out.push_back(blk);
        if (!blk.closed) break;
        p = e + kEnd.size();
    }
    return out;
}

// The reply as tierlm shows it: "<think>{thinking}</think>\n\n{answer}". Grows only at the end as the raw reply does.
// Tool calls aren't shown (they're returned separately).
inline std::string display(const std::string& raw) {
    std::string s;
    bool text_before = false;                                               // (blocks of text: a blank line between)
    for (const Block& b : blocks_of(raw)) {
        if (b.kind == "thinking") { s += "<think>" + b.text; if (b.closed) s += "</think>\n\n"; }
        else if (b.kind == "text") { s += (text_before ? "\n\n" : "") + b.text; text_before = true; }
    }
    return s;
}

// The tool calls in a reply, like agent::parse_tool_calls
inline agent::Parsed parse(const std::string& raw, const ojson& tools) {
    (void)tools;
    agent::Parsed r;
    r.content = display(raw);
    for (const Block& b : blocks_of(raw)) {
        if (b.kind != "tool") continue;
        const json j = json::parse(agent::trim(b.text), nullptr, false);
        json args = j.is_object() && j.contains("args") && j["args"].is_object() ? j["args"] : json::object();
        const std::string name = !b.name.empty() ? b.name : j.is_object() ? j.value("name", std::string()) : "";
        if (name.empty()) continue;
        r.calls.push_back({{"name", name}, {"arguments", args}, {"arguments_json", compact(args)}});
    }
    return r;
}

}  // namespace inkling
