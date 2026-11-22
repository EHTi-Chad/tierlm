// What apps send as a conversation (OpenAI's chat messages), made plain for tierlm's chat formats. For tierlm-server
// and the chat-format tests.
#pragma once

#include "agent.h"

#include <string>

namespace tierlm_msgs {

using json = nlohmann::json;

// Images in a conversation. A message may carry "images" (data: URLs or plain base64 of PNG / JPEG / ...); in its
// text, each image's place is kPictureMark (the page puts them before the text; the API where they were sent).
// For Qwen3.5 / 3.6 the mark is what their chat template writes for an image; its <|image_pad|> token is replaced by
// the image's words just before the model reads the prompt.
inline const char* const kPictureMark = "<|vision_start|><|image_pad|><|vision_end|>";

// OpenAI messages to plain {role, content} ones. Content may be a string, a list of parts
// ([{"type": "text", "text": ...}], images are skipped: no vision yet) or null. "developer" is OpenAI's newer
// name for "system". Tool calls ({"function": {"name", "arguments": "<json>"}}) are kept as {"name", "arguments": {...}},
// and tool results as role "tool".
inline json normalize_messages(const json& in) {
    json out = json::array();
    for (const auto& m : in) {
        if (!m.is_object()) continue;
        std::string role = m.value("role", std::string("user")), text;
        const json c = m.contains("content") ? m["content"] : json();
        json images = json::array();
        if (c.is_string()) text = c.get<std::string>();
        else if (c.is_array())
            for (const auto& part : c) {
                if (part.is_string()) { text += part.get<std::string>(); continue; }
                if (!part.is_object()) continue;
                const std::string type = part.value("type", std::string());
                if (type == "text" || type == "input_text") text += part.value("text", std::string());
                else if (type == "image_url" || type == "input_image" || type == "image") {
                    // {"image_url": {"url": "data:..."}} (OpenAI chat), {"image_url": "data:..."} (OpenAI responses),
                    // {"source": {"data": "...", "media_type": ...}} (Anthropic-style)
                    std::string url;
                    const json iu = part.contains("image_url") ? part["image_url"] : json();
                    if (iu.is_object()) url = iu.value("url", std::string());
                    else if (iu.is_string()) url = iu.get<std::string>();
                    else if (part.contains("source") && part["source"].is_object()) url = part["source"].value("data", std::string());
                    else if (part.contains("data") && part["data"].is_string()) url = part["data"].get<std::string>();
                    if (url.empty()) continue;
                    images.push_back(url);
                    text += kPictureMark;                          // where the image goes
                }
            }
        if (role == "developer") role = "system";
        if (role == "function") role = "tool";
        if (role != "system" && role != "user" && role != "assistant" && role != "tool") role = "user";
        // Thinking sent back separately (reasoning_content) goes back where the model wrote it.
        if (role == "assistant" && m.contains("reasoning_content") && m["reasoning_content"].is_string() &&
            !m["reasoning_content"].get<std::string>().empty() && text.find("<think>") == std::string::npos)
            text = "<think>\n" + m["reasoning_content"].get<std::string>() + "\n</think>\n\n" + text;
        json o = {{"role", role}, {"content", text}};
        if (!images.empty()) o["images"] = images;
        if (role == "assistant" && m.contains("tool_calls") && m["tool_calls"].is_array()) {
            json calls = json::array();
            for (const auto& tc : m["tool_calls"]) {
                const json f = agent::tool_function(tc);
                if (!f.is_object()) continue;
                json args = f.contains("arguments") ? f["arguments"] : json::object();
                json call = {{"name", f.value("name", std::string())}};
                if (args.is_string()) {                            // as sent: keeps the order the model wrote them in
                    call["arguments_json"] = args;
                    json p = json::parse(args.get<std::string>(), nullptr, false);
                    args = p.is_object() ? p : json::object();
                }
                call["arguments"] = args.is_object() ? args : json::object();
                calls.push_back(call);
            }
            if (!calls.empty()) o["tool_calls"] = calls;
        }
        out.push_back(o);
    }
    return out;
}

}  // namespace tierlm_msgs
