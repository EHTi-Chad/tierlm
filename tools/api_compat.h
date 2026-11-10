// tierlm - two more ways for apps to talk to the model, beside OpenAI's chat API (/v1/chat/completions):
//   POST /v1/messages   Anthropic's Messages API (Claude Code and apps built on Anthropic's SDK)
//   POST /v1/responses  OpenAI's Responses API (Codex and apps built on OpenAI's newer SDK calls)
//
// Both are translated to and from what the chat API already works with: a list of OpenAI-style messages and
// function tools going in; the reply's thinking, its text and its tool calls coming out. This file has the
// translation of the requests (no engine in it, so it can be tested alone); the handlers in server.cpp write the
// replies.
#pragma once

#include "agent.h"

#include <string>
#include <vector>

namespace compat {

using json = nlohmann::json;
using agent::ojson;

// What either request asks for, in the chat API's terms.
struct Request {
    json messages = json::array();        // OpenAI chat messages (normalize_messages reads them)
    ojson tools = ojson::array();         // OpenAI function tools, in the order the app wrote them
    json think = json::object();          // what take_think reads: {"thinking": bool} / {"reasoning_effort": "..."}
    int max_tokens = -1;
    std::vector<std::string> stops;
    bool stream = false;
    std::string error;                    // not empty: the request can't be served, and why
};

// A list of content blocks as plain text (text blocks joined; other kinds left out).
inline std::string blocks_text(const json& c) {
    if (c.is_string()) return c.get<std::string>();
    std::string s;
    if (c.is_array())
        for (const auto& b : c) {
            if (b.is_string()) { s += b.get<std::string>(); continue; }
            if (!b.is_object()) continue;
            const std::string t = b.value("type", std::string());
            if ((t == "text" || t == "input_text" || t == "output_text" || t == "summary_text" || t == "reasoning_text") && b.contains("text") && b["text"].is_string())
                s += (s.empty() ? "" : "\n") + b["text"].get<std::string>();
        }
    return s;
}

inline ojson function_tool(const ojson& name, const ojson& description, const ojson& parameters) {
    ojson f = ojson::object();
    f["name"] = name;
    if (description.is_string()) f["description"] = description;
    f["parameters"] = parameters.is_object() ? parameters : ojson::parse("{\"type\":\"object\",\"properties\":{}}");
    ojson t = ojson::object();
    t["type"] = "function";
    t["function"] = f;
    return t;
}

// An assistant tool call in OpenAI's chat form.
inline json chat_call(const std::string& id, const std::string& name, const std::string& arguments_json) {
    return {{"id", id}, {"type", "function"}, {"function", {{"name", name}, {"arguments", arguments_json}}}};
}

// ---- Anthropic's Messages API ----
// {"model", "max_tokens", "system": "..." | [blocks], "messages": [{"role", "content": "..." | [blocks]}],
//  "tools": [{"name", "description", "input_schema"}], "tool_choice", "stop_sequences", "thinking", "stream"}
// Blocks: text, image (base64), tool_use (the model's call), tool_result (the app's answer), thinking.
inline Request from_anthropic(const std::string& raw) {
    Request r;
    const json body = json::parse(raw, nullptr, false);
    if (!body.is_object() || !body.contains("messages") || !body["messages"].is_array()) { r.error = "expected a JSON body with a \"messages\" array"; return r; }
    if (body.contains("system")) {
        const std::string sys = blocks_text(body["system"]);
        if (!sys.empty()) r.messages.push_back({{"role", "system"}, {"content", sys}});
    }
    for (const auto& m : body["messages"]) {
        if (!m.is_object()) continue;
        const std::string role = m.value("role", std::string("user"));
        const json c = m.contains("content") ? m["content"] : json();
        if (c.is_string()) { r.messages.push_back({{"role", role}, {"content", c}}); continue; }
        if (!c.is_array()) continue;
        if (role == "assistant") {
            std::string text, thinking;
            json calls = json::array();
            for (const auto& b : c) {
                if (!b.is_object()) continue;
                const std::string t = b.value("type", std::string());
                if (t == "text") text += b.value("text", std::string());
                else if (t == "thinking") thinking += b.value("thinking", std::string());
                else if (t == "tool_use")
                    calls.push_back(chat_call(b.value("id", std::string()), b.value("name", std::string()),
                                              (b.contains("input") && b["input"].is_object() ? b["input"] : json::object()).dump()));
            }
            json o = {{"role", "assistant"}, {"content", text}};
            if (!thinking.empty()) o["reasoning_content"] = thinking;
            if (!calls.empty()) o["tool_calls"] = calls;
            r.messages.push_back(o);
            continue;
        }
        // A user turn: the results of the tools the model called come first (each its own "tool" message), then
        // whatever the user wrote, with its images.
        json parts = json::array();
        for (const auto& b : c) {
            if (b.is_string()) { parts.push_back({{"type", "text"}, {"text", b}}); continue; }
            if (!b.is_object()) continue;
            const std::string t = b.value("type", std::string());
            if (t == "tool_result") {
                std::string out = b.contains("content") ? blocks_text(b["content"]) : std::string();
                if (b.value("is_error", false)) out = "Error: " + out;
                r.messages.push_back({{"role", "tool"}, {"tool_call_id", b.value("tool_use_id", std::string())}, {"content", out}});
            } else if (t == "text") parts.push_back({{"type", "text"}, {"text", b.value("text", std::string())}});
            else if (t == "image" && b.contains("source") && b["source"].is_object() && b["source"].value("type", std::string()) == "base64")
                parts.push_back({{"type", "image_url"}, {"image_url", {{"url", "data:" + b["source"].value("media_type", std::string("image/png")) +
                                                                               ";base64," + b["source"].value("data", std::string())}}}});
        }
        if (!parts.empty()) r.messages.push_back({{"role", "user"}, {"content", parts}});
    }
    const ojson ob = ojson::parse(raw, nullptr, false);
    const bool no_tools = body.contains("tool_choice") && body["tool_choice"].is_object() && body["tool_choice"].value("type", std::string()) == "none";
    if (!no_tools && ob.is_object() && ob.contains("tools") && ob["tools"].is_array())
        for (const auto& t : ob["tools"])
            if (t.is_object() && t.contains("name") && t["name"].is_string() && (!t.contains("type") || t["type"] == "custom"))
                r.tools.push_back(function_tool(t["name"], t.contains("description") ? t["description"] : ojson(),
                                                t.contains("input_schema") ? t["input_schema"] : ojson()));
    if (body.contains("max_tokens") && body["max_tokens"].is_number()) r.max_tokens = body["max_tokens"].get<int>();
    if (body.contains("stop_sequences") && body["stop_sequences"].is_array())
        for (const auto& s : body["stop_sequences"]) if (s.is_string()) r.stops.push_back(s.get<std::string>());
    if (body.contains("thinking") && body["thinking"].is_object()) {
        const std::string t = body["thinking"].value("type", std::string());
        if (t == "enabled" || t == "adaptive") r.think["thinking"] = true;
        else if (t == "disabled") r.think["thinking"] = false;
    }
    r.stream = body.value("stream", false);
    return r;
}

// ---- OpenAI's Responses API ----
// {"model", "instructions", "input": "..." | [items], "tools": [{"type": "function", "name", "description",
//  "parameters"}], "tool_choice", "max_output_tokens", "reasoning": {"effort"}, "stream"}
// Items: messages ({"role", "content": "..." | [input_text / input_image / output_text parts]}), function_call
// (the model's call), function_call_output (the app's answer), reasoning.
inline Request from_responses(const std::string& raw) {
    Request r;
    const json body = json::parse(raw, nullptr, false);
    if (!body.is_object() || !body.contains("input")) { r.error = "expected a JSON body with \"input\""; return r; }
    if (body.contains("previous_response_id") && body["previous_response_id"].is_string() && !body["previous_response_id"].get<std::string>().empty()) {
        r.error = "previous_response_id isn't supported: tierlm doesn't keep responses, send the conversation in \"input\"";
        return r;
    }
    if (body.contains("instructions") && body["instructions"].is_string() && !body["instructions"].get<std::string>().empty())
        r.messages.push_back({{"role", "system"}, {"content", body["instructions"]}});
    const json in = body["input"];
    if (in.is_string()) r.messages.push_back({{"role", "user"}, {"content", in}});
    else if (in.is_array()) {
        std::string thinking;                              // a reasoning item belongs to the assistant turn after it
        auto assistant = [&]() -> json& {                  // the assistant message being put together (calls join it)
            if (r.messages.empty() || r.messages.back().value("role", std::string()) != "assistant")
                r.messages.push_back({{"role", "assistant"}, {"content", ""}});
            json& a = r.messages.back();
            if (!thinking.empty()) { a["reasoning_content"] = thinking; thinking.clear(); }
            return a;
        };
        for (const auto& it : in) {
            if (it.is_string()) { r.messages.push_back({{"role", "user"}, {"content", it}}); continue; }
            if (!it.is_object()) continue;
            const std::string type = it.value("type", std::string(it.contains("role") ? "message" : ""));
            if (type == "message") {
                const std::string role = it.value("role", std::string("user"));
                const json c = it.contains("content") ? it["content"] : json();
                if (role == "assistant") {
                    json& a = assistant();
                    a["content"] = a.value("content", std::string()) + blocks_text(c);
                } else if (c.is_string()) r.messages.push_back({{"role", role}, {"content", c}});
                else {
                    json parts = json::array();
                    if (c.is_array())
                        for (const auto& p : c) {
                            if (!p.is_object()) continue;
                            const std::string pt = p.value("type", std::string());
                            if (pt == "input_text" || pt == "text") parts.push_back({{"type", "text"}, {"text", p.value("text", std::string())}});
                            else if (pt == "input_image" && p.contains("image_url") && p["image_url"].is_string())
                                parts.push_back({{"type", "image_url"}, {"image_url", {{"url", p["image_url"]}}}});
                        }
                    r.messages.push_back({{"role", role}, {"content", parts}});
                }
            } else if (type == "function_call") {
                json& a = assistant();
                if (!a.contains("tool_calls")) a["tool_calls"] = json::array();
                const json args = it.contains("arguments") ? it["arguments"] : json("{}");
                a["tool_calls"].push_back(chat_call(it.value("call_id", std::string()), it.value("name", std::string()),
                                                    args.is_string() ? args.get<std::string>() : args.dump()));
            } else if (type == "function_call_output") {
                const json out = it.contains("output") ? it["output"] : json("");
                r.messages.push_back({{"role", "tool"}, {"tool_call_id", it.value("call_id", std::string())},
                                      {"content", out.is_string() ? out.get<std::string>() : blocks_text(out)}});
            } else if (type == "reasoning") {
                std::string t = it.contains("content") ? blocks_text(it["content"]) : std::string();
                if (t.empty() && it.contains("summary")) t = blocks_text(it["summary"]);
                thinking += t;
            }
        }
    } else { r.error = "\"input\" must be text or a list of items"; return r; }
    const ojson ob = ojson::parse(raw, nullptr, false);
    const bool no_tools = body.contains("tool_choice") && body["tool_choice"].is_string() && body["tool_choice"].get<std::string>() == "none";
    if (!no_tools && ob.is_object() && ob.contains("tools") && ob["tools"].is_array())
        for (const auto& t : ob["tools"])
            if (t.is_object() && t.contains("type") && t["type"] == "function" && t.contains("name") && t["name"].is_string())
                r.tools.push_back(function_tool(t["name"], t.contains("description") ? t["description"] : ojson(),
                                                t.contains("parameters") ? t["parameters"] : ojson()));
    if (body.contains("max_output_tokens") && body["max_output_tokens"].is_number()) r.max_tokens = body["max_output_tokens"].get<int>();
    if (body.contains("reasoning") && body["reasoning"].is_object() && body["reasoning"].contains("effort") && body["reasoning"]["effort"].is_string())
        r.think["reasoning_effort"] = body["reasoning"]["effort"];
    r.stream = body.value("stream", false);
    return r;
}

}
