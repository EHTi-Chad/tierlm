// Saved chats as files: one folder per chat in a "chats" folder next to tierlm's settings, holding
//   chat.json   everything the page needs to show the chat again (messages, which ones are archived, the summary)
//   chat.md     the same conversation as a readable Markdown file (Notepad, VS Code, Obsidian...), with links to
//               the files that were attached (copied into files/) and to the workspace files the model worked on
//   files/      the attached files
// The folder is named "<date> <title> (<id>)", so it reads well in Explorer and is still found by its id.
//
// Also the archive's look-back: when a long conversation no longer fits the context, its oldest messages are
// archived (kept here, no longer sent to the model). For each new message, recall() finds the archived passages
// that best match it (by meaning with an embedding model, else by keywords), and those go back to the model.
#pragma once

#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace chats {

using json = nlohmann::json;
namespace fs = std::filesystem;

struct Error : std::runtime_error { using std::runtime_error::runtime_error; };

// Chat ids are made by the page (base-36 time): letters and digits only, so they are safe in a folder name.
inline bool valid_id(const std::string& id) {
    if (id.empty() || id.size() > 32) return false;
    for (unsigned char c : id) if (!std::isalnum(c)) return false;
    return true;
}

// A title or file name made safe for Windows folder and file names (no \ / : * ? " < > |, no control characters,
// no trailing dots or spaces, not too long).
inline std::string clean_name(const std::string& s, size_t max_len = 60) {
    std::string r;
    for (unsigned char c : s) {
        if (c < 32 || std::string("\\/:*?\"<>|").find((char)c) != std::string::npos) r += ' ';
        else r += (char)c;
    }
    std::string out;
    for (char c : r) if (!(c == ' ' && (out.empty() || out.back() == ' '))) out += c;   // single spaces
    if (out.size() > max_len) {
        out.resize(max_len);
        while (!out.empty() && ((unsigned char)out.back() & 0xC0) == 0x80) out.pop_back();   // whole UTF-8 characters
        if (!out.empty() && ((unsigned char)out.back() & 0xC0) == 0xC0) out.pop_back();
    }
    while (!out.empty() && (out.back() == ' ' || out.back() == '.')) out.pop_back();
    while (!out.empty() && out.front() == ' ') out.erase(out.begin());
    return out.empty() ? "chat" : out;
}

inline std::string local_time(long long ms, const char* fmt) {
    std::time_t t = (std::time_t)(ms / 1000);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof buf, fmt, &tm);
    return buf;
}

inline std::string read_all(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Written to a temporary file first, then moved over the old one, so a crash never leaves half a chat.
inline void write_all(const fs::path& p, const std::string& s) {
    fs::path tmp = p;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw Error("can't write " + p.u8string());
        f.write(s.data(), (std::streamsize)s.size());
        if (!f) throw Error("can't write " + p.u8string());
    }
    std::error_code ec;
    fs::rename(tmp, p, ec);
    if (ec) { fs::remove(p, ec); fs::rename(tmp, p, ec); }
    if (ec) throw Error("can't write " + p.u8string() + ": " + ec.message());
}

// The folder of chat `id`, or an empty path.
inline fs::path folder_of(const fs::path& dir, const std::string& id) {
    std::error_code ec;
    if (!valid_id(id) || !fs::is_directory(dir, ec)) return {};
    const std::string tail = "(" + id + ")";
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_directory(ec)) continue;
        const std::string n = e.path().filename().u8string();
        if (n == id || (n.size() >= tail.size() && n.compare(n.size() - tail.size(), tail.size(), tail) == 0)) return e.path();
    }
    return {};
}

inline json list(const fs::path& dir) {
    json out = json::array();
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return out;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_directory(ec)) continue;
        json c = json::parse(read_all(e.path() / "chat.json"), nullptr, false);
        if (!c.is_object() || !c.contains("id")) continue;
        int n = 0, archived = 0;
        for (const auto& m : c.value("messages", json::array())) { ++n; if (m.value("archived", false)) ++archived; }
        out.push_back({{"id", c["id"]}, {"title", c.value("title", std::string("Chat"))}, {"created", c.value("created", 0LL)},
                       {"updated", c.value("updated", 0LL)}, {"model", c.value("model", std::string())}, {"messages", n},
                       {"archived", archived}, {"folder", e.path().filename().u8string()}});
        if (c.contains("workspace") && c["workspace"].is_string()) out.back()["workspace"] = c["workspace"];
    }
    std::sort(out.begin(), out.end(), [](const json& a, const json& b) { return a.value("updated", 0LL) > b.value("updated", 0LL); });
    return out;
}

inline json load(const fs::path& dir, const std::string& id) {
    const fs::path f = folder_of(dir, id);
    if (f.empty()) throw Error("no such chat");
    json c = json::parse(read_all(f / "chat.json"), nullptr, false);
    if (!c.is_object()) throw Error("the chat file is damaged");
    return c;
}

// ---- the Markdown version ----
inline std::string file_url(const fs::path& p) {                 // file:///C:/My%20Project/a.cpp
    std::string s = p.generic_u8string(), r = "file:///";
    if (!s.empty() && s[0] == '/') s.erase(0, 1);
    for (unsigned char c : s) {
        if (std::isalnum(c) || std::string("/-_.~:").find((char)c) != std::string::npos || c >= 0x80) r += (char)c;
        else { char b[4]; std::snprintf(b, sizeof b, "%%%02X", c); r += b; }
    }
    return r;
}
inline std::string rel_url(const std::string& s) {               // files/my%20notes.txt
    std::string r;
    for (unsigned char c : s) {
        if (std::isalnum(c) || std::string("/-_.~").find((char)c) != std::string::npos || c >= 0x80) r += (char)c;
        else { char b[4]; std::snprintf(b, sizeof b, "%%%02X", c); r += b; }
    }
    return r;
}
// The longest run of backticks in s, plus one: a fence that can't be closed by the text inside.
inline std::string fence_for(const std::string& s) {
    size_t best = 0, run = 0;
    for (char c : s) { run = c == '`' ? run + 1 : 0; best = std::max(best, run); }
    return std::string(std::max<size_t>(3, best + 1), '`');
}
// <think>...</think> as a collapsed "Thinking" section.
inline std::string thinking_collapsed(std::string s) {
    for (size_t a; (a = s.find("<think>")) != std::string::npos;) {
        size_t b = s.find("</think>", a);
        const std::string inner = s.substr(a + 7, b == std::string::npos ? std::string::npos : b - a - 7);
        const bool empty = inner.find_first_not_of(" \n\r\t") == std::string::npos;
        const std::string rep = empty ? std::string() : "<details><summary>Thinking</summary>\n\n" + inner + "\n\n</details>\n\n";
        s.replace(a, b == std::string::npos ? std::string::npos : b + 8 - a, rep);
    }
    return s;
}
inline std::string text_of(const json& content) {
    if (content.is_string()) return content.get<std::string>();
    if (content.is_array()) {                                     // OpenAI parts: the text ones
        std::string r;
        for (const auto& p : content) if (p.is_object() && p.value("type", std::string()) == "text") r += p.value("text", std::string());
        return r;
    }
    return {};
}

// Images sent in a chat are kept in it (as data: URLs, to send again with each turn) and also as files next to
// chat.md, so it shows them: files/image-<fingerprint>.png (or .jpg ...).
inline std::string picture_file_name(const std::string& url) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char ch : url) { h ^= ch; h *= 1099511628211ULL; }
    std::string ext = "png";
    const size_t slash = url.find("image/"), semi = url.find(';');
    if (url.rfind("data:", 0) == 0 && slash != std::string::npos && semi != std::string::npos && slash < semi) {
        ext = url.substr(slash + 6, semi - slash - 6);
        if (ext == "jpeg") ext = "jpg";
        if (ext.empty() || ext.size() > 5 || ext.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789") != std::string::npos) ext = "png";
    }
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", (unsigned long long)h);
    return std::string("image-") + std::string(hex).substr(0, 12) + "." + ext;
}
inline std::string picture_bytes_of(const std::string& url) {    // base64 of a data: URL to bytes ("" if not one)
    const size_t comma = url.find(',');
    if (url.rfind("data:", 0) != 0 || comma == std::string::npos) return {};
    std::string out;
    uint32_t acc = 0; int bits = 0;
    for (size_t i = comma + 1; i < url.size(); ++i) {
        const unsigned char ch = (unsigned char)url[i];
        int v = ch >= 'A' && ch <= 'Z' ? ch - 'A' : ch >= 'a' && ch <= 'z' ? ch - 'a' + 26 : ch >= '0' && ch <= '9' ? ch - '0' + 52
              : ch == '+' || ch == '-' ? 62 : ch == '/' || ch == '_' ? 63 : -1;
        if (v < 0) continue;
        acc = acc << 6 | (uint32_t)v; bits += 6;
        if (bits >= 8) { bits -= 8; out += (char)(acc >> bits & 0xFF); }
    }
    return out;
}

inline std::string markdown(const json& c, const std::string& ws_folder) {
    std::ostringstream md;
    const std::string title = c.value("title", std::string("Chat"));
    md << "# " << title << "\n\n";
    md << "*" << (c.value("model", std::string()).empty() ? std::string("") : c.value("model", std::string()) + " · ")
       << "started " << local_time(c.value("created", 0LL), "%Y-%m-%d %H:%M") << " · saved by tierlm*\n\n";
    if (c.contains("workspace") && c["workspace"].is_string() && !c["workspace"].get<std::string>().empty())
        md << "*Project folder: [" << c["workspace"].get<std::string>() << "](" << file_url(fs::u8path(c["workspace"].get<std::string>())) << ")*\n\n";
    if (!c.value("summary", std::string()).empty()) {
        std::istringstream sum(c.value("summary", std::string()));
        md << "> **Summary of the archived part, given to the model:**\n>\n";
        for (std::string l; std::getline(sum, l);) md << "> " << l << "\n";
        md << "\n";
    }
    md << "---\n\n";
    const json& ms = c.contains("messages") && c["messages"].is_array() ? c["messages"] : json::array();
    bool in_archive = !ms.empty() && ms[0].value("archived", false);
    if (in_archive) md << "*The messages down to the next line were archived from the model's memory: they stay here, and tierlm looks back at them when they're relevant.*\n\n";
    const fs::path ws = ws_folder.empty() ? fs::path() : fs::u8path(ws_folder);
    auto ws_link = [&](const std::string& rel) {
        if (rel.empty()) return std::string("the workspace");
        return ws.empty() ? "`" + rel + "`" : "[" + rel + "](" + file_url(ws / fs::u8path(rel)) + ")";
    };
    for (const auto& m : ms) {
        if (!m.is_object()) continue;
        if (in_archive && !m.value("archived", false)) {
            in_archive = false;
            md << "---\n\n*The messages above were archived from the model's memory.*\n\n---\n\n";
        }
        const std::string role = m.value("role", std::string());
        const long long t = m.value("time", 0LL);
        const std::string when = t ? " · " + local_time(t, "%H:%M") : "";
        if (role == "user") {
            const std::string shown = m.contains("shown") ? text_of(m["shown"]) : text_of(m.value("content", json()));
            md << "**You**" << when << "\n\n";
            for (const auto& u : m.value("images", json::array()))
                if (u.is_string()) md << "![image](" << rel_url("files/" + picture_file_name(u.get<std::string>())) << ")\n\n";
            md << shown << "\n\n";
            for (const auto& f : m.value("files", json::array())) {
                const std::string n = clean_name(f.value("name", std::string("file")), 120);
                md << "📎 [" << n << "](" << rel_url("files/" + n) << ")\n\n";
            }
        } else if (role == "assistant") {
            std::string text = thinking_collapsed(text_of(m.value("content", json())));
            while (!text.empty() && (text[0] == '\n' || text[0] == ' ')) text.erase(0, 1);
            md << "**" << (c.value("model", std::string()).empty() ? "Model" : c.value("model", std::string())) << "**" << when << "\n\n";
            if (!text.empty()) md << text << "\n\n";
            for (const auto& call : m.value("tool_calls", json::array())) {
                const std::string name = call.value("name", std::string());
                json args = call.contains("arguments") ? call["arguments"] : json::object();
                if (args.is_string()) args = json::parse(args.get<std::string>(), nullptr, false);
                const std::string path = args.is_object() ? args.value("path", std::string()) : std::string();
                if (name == "read_file") md << "🔧 Read " << ws_link(path) << "\n\n";
                else if (name == "list_files") md << "🔧 Listed " << ws_link(path.empty() ? "." : path) << "\n\n";
                else if (name == "search_files") md << "🔧 Searched the workspace for `" << (args.is_object() ? args.value("query", std::string()) : "") << "`\n\n";
                else if (name == "write_file" || name == "edit_file") {
                    md << "🔧 Proposed a change to " << ws_link(path) << "\n\n";
                    if (args.is_object() && name == "edit_file") {
                        std::string d;
                        std::istringstream o(args.value("old_text", std::string())), n(args.value("new_text", std::string()));
                        for (std::string l; std::getline(o, l);) d += "- " + l + "\n";
                        for (std::string l; std::getline(n, l);) d += "+ " + l + "\n";
                        const std::string fe = fence_for(d);
                        md << fe << "diff\n" << d << fe << "\n\n";
                    } else if (args.is_object()) {
                        const std::string body = args.value("content", std::string());
                        const std::string fe = fence_for(body);
                        md << "<details><summary>New content</summary>\n\n" << fe << "\n" << body << "\n" << fe << "\n\n</details>\n\n";
                    }
                } else md << "🔧 " << name << "\n\n";
            }
        } else if (role == "tool") {
            std::string r = text_of(m.value("content", json()));
            const std::string first = r.substr(0, r.find('\n'));
            if (r.size() > 400 || r.find('\n') != std::string::npos) {
                const std::string fe = fence_for(r);
                md << "<details><summary>Result: " << clean_name(first, 80) << "</summary>\n\n" << fe << "\n" << r << "\n" << fe << "\n\n</details>\n\n";
            } else md << "↳ " << r << "\n\n";
        }
    }
    return md.str();
}

inline std::string safe_file_name(const std::string& n) {
    std::string base = n;
    const size_t s = base.find_last_of("/\\");
    if (s != std::string::npos) base = base.substr(s + 1);
    return clean_name(base, 120);
}

// Saves chat.json and chat.md (making the folder the first time). Returns the folder.
inline fs::path save(const fs::path& dir, const std::string& id, json c, const std::string& ws_folder) {
    if (!valid_id(id)) throw Error("bad chat id");
    if (!c.is_object()) throw Error("expected a chat object");
    c["id"] = id;
    std::error_code ec;
    fs::create_directories(dir, ec);
    fs::path f = folder_of(dir, id);
    if (f.empty()) {
        const long long created = c.value("created", 0LL);
        f = dir / fs::u8path((created ? local_time(created, "%Y-%m-%d") + " " : std::string()) +
                             clean_name(c.value("title", std::string("Chat")), 50) + " (" + id + ")");
        fs::create_directories(f, ec);
        if (ec) throw Error("can't create " + f.u8string() + ": " + ec.message());
    }
    write_all(f / "chat.json", c.dump(1, ' ', false, json::error_handler_t::replace));
    for (const auto& m : c.value("messages", json::array()))                 // its images, as files (once)
        for (const auto& u : m.value("images", json::array())) {
            if (!u.is_string()) continue;
            const fs::path pf = f / "files" / fs::u8path(picture_file_name(u.get<std::string>()));
            if (fs::exists(pf, ec)) continue;
            const std::string bytes = picture_bytes_of(u.get<std::string>());
            if (bytes.empty()) continue;
            fs::create_directories(f / "files", ec);
            write_all(pf, bytes);
        }
    write_all(f / "chat.md", markdown(c, ws_folder));
    return f;
}

inline void remove(const fs::path& dir, const std::string& id) {
    const fs::path f = folder_of(dir, id);
    if (f.empty()) return;
    std::error_code ec;
    fs::remove_all(f, ec);
    if (ec) throw Error("can't delete " + f.u8string() + ": " + ec.message());
}

inline void put_file(const fs::path& dir, const std::string& id, const std::string& name, const std::string& data) {
    const fs::path f = folder_of(dir, id);
    if (f.empty()) throw Error("no such chat");
    std::error_code ec;
    fs::create_directories(f / "files", ec);
    write_all(f / "files" / fs::u8path(safe_file_name(name)), data);
}

inline std::string get_file(const fs::path& dir, const std::string& id, const std::string& name, bool& found) {
    const fs::path f = folder_of(dir, id);
    found = false;
    if (f.empty()) return {};
    const fs::path p = f / "files" / fs::u8path(safe_file_name(name));
    std::error_code ec;
    if (!fs::is_regular_file(p, ec)) return {};
    found = true;
    return read_all(p);
}

// ---- looking back at the archive ----
struct Passage { int msg; std::string role, text; long long time; double score = 0; };

inline std::vector<std::string> words(const std::string& s) {
    static const std::set<std::string> stop = {"the", "and", "for", "that", "this", "with", "you", "are", "was", "but", "not", "have",
        "has", "had", "can", "could", "would", "should", "what", "when", "where", "which", "who", "how", "why", "its", "it's", "from",
        "they", "them", "then", "than", "there", "their", "will", "just", "about", "into", "your", "our", "out", "all", "any", "also",
        "one", "some", "like", "did", "does", "doing", "been", "being", "were", "is", "it", "to", "of", "in", "on", "a", "an", "or",
        "be", "as", "at", "by", "we", "i", "me", "my", "so", "if", "do", "no", "yes", "up"};
    std::vector<std::string> out;
    std::string w;
    auto flush = [&] { if (w.size() >= 2 && !stop.count(w)) out.push_back(w); w.clear(); };
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '_' || c >= 0x80) w += (char)std::tolower(c);
        else flush();
    }
    flush();
    return out;
}

// The archived messages of chat c, cut into passages of about 1,200 characters (on line breaks where possible).
inline std::vector<Passage> archived_passages(const json& c) {
    std::vector<Passage> ps;
    const json& ms = c.contains("messages") && c["messages"].is_array() ? c["messages"] : json::array();
    for (int i = 0; i < (int)ms.size(); ++i) {
        const json& m = ms[i];
        if (!m.is_object() || !m.value("archived", false)) continue;
        const std::string role = m.value("role", std::string());
        if (role != "user" && role != "assistant" && role != "tool") continue;
        std::string t = role == "user" && m.contains("shown") ? text_of(m["shown"]) : text_of(m.value("content", json()));
        for (size_t a; (a = t.find("<think>")) != std::string::npos;) {      // the model's reasoning isn't recalled
            size_t b = t.find("</think>", a);
            t.erase(a, b == std::string::npos ? std::string::npos : b + 8 - a);
        }
        for (size_t at = 0; at < t.size();) {
            size_t end = std::min(t.size(), at + 1200);
            if (end < t.size()) { size_t nl = t.rfind('\n', end); if (nl != std::string::npos && nl > at + 600) end = nl + 1; }
            std::string part = t.substr(at, end - at);
            if (part.find_first_not_of(" \n\r\t") != std::string::npos) ps.push_back({i, role, part, m.value("time", 0LL)});
            at = end;
        }
    }
    return ps;
}

// The best few passages for `query` (at most `max_chars` of text in all), best first. With `embed` (an embedding
// model: text -> unit vector, is_query), by meaning; otherwise by keywords (BM25).
inline std::vector<Passage> recall(const json& c, const std::string& query, size_t max_chars = 6000, int max_n = 4,
                                   const std::function<std::vector<float>(const std::string&, bool)>& embed = nullptr) {
    std::vector<Passage> ps = archived_passages(c);
    if (ps.empty() || query.empty()) return {};
    bool scored = false;
    if (embed) {
        // Embeddings of passages are kept between messages (the archive changes rarely).
        static std::unordered_map<std::string, std::vector<float>> cache;
        if (cache.size() > 20000) cache.clear();
        const std::vector<float> q = embed(query, true);
        if (!q.empty()) {
            for (auto& p : ps) {
                auto it = cache.find(p.text);
                if (it == cache.end()) it = cache.emplace(p.text, embed(p.text, false)).first;
                const auto& v = it->second;
                double d = 0;
                for (size_t k = 0; k < v.size() && k < q.size(); ++k) d += (double)v[k] * q[k];
                p.score = d;
            }
            scored = true;
            // Only clearly related passages (cosine similarity of these models: unrelated text ~0.3-0.5).
            ps.erase(std::remove_if(ps.begin(), ps.end(), [](const Passage& p) { return p.score < 0.55; }), ps.end());
        }
    }
    if (!scored) {
        const std::vector<std::string> qw = words(query);
        if (qw.empty()) return {};
        std::vector<std::unordered_map<std::string, int>> tf(ps.size());
        std::unordered_map<std::string, int> df;
        double avg = 0;
        for (size_t i = 0; i < ps.size(); ++i) {
            const auto w = words(ps[i].text);
            avg += (double)w.size();
            for (const auto& x : w) tf[i][x]++;
            for (const auto& [x, n] : tf[i]) df[x]++;
        }
        avg = std::max(1.0, avg / ps.size());
        const double N = (double)ps.size(), k1 = 1.2, b = 0.75;
        const std::set<std::string> qset(qw.begin(), qw.end());
        for (size_t i = 0; i < ps.size(); ++i) {
            double len = 0;
            for (const auto& [x, n] : tf[i]) len += n;
            double s = 0;
            for (const auto& x : qset) {
                auto it = tf[i].find(x);
                if (it == tf[i].end()) continue;
                const double idf = std::log(1.0 + (N - df[x] + 0.5) / (df[x] + 0.5));
                s += idf * it->second * (k1 + 1) / (it->second + k1 * (1 - b + b * len / avg));
            }
            ps[i].score = s;
        }
        ps.erase(std::remove_if(ps.begin(), ps.end(), [](const Passage& p) { return p.score < 1.0; }), ps.end());
    }
    std::stable_sort(ps.begin(), ps.end(), [](const Passage& a, const Passage& b) { return a.score > b.score; });
    if (!ps.empty() && !scored) {                     // keywords: only passages close to the best match
        const double best = ps[0].score;
        ps.erase(std::remove_if(ps.begin(), ps.end(), [&](const Passage& p) { return p.score < 0.5 * best; }), ps.end());
    }
    std::vector<Passage> out;
    size_t used = 0;
    for (const auto& p : ps) {
        if ((int)out.size() >= max_n || used + p.text.size() > max_chars) break;
        out.push_back(p);
        used += p.text.size();
    }
    std::sort(out.begin(), out.end(), [](const Passage& a, const Passage& b) { return a.msg < b.msg; });   // in order
    return out;
}

}  // namespace chats
