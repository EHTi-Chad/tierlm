// The workspace: one folder on the computer running tierlm that the model can work in, from tierlm's own chat page.
//
// The model gets five tools (tool calling, see agent.h): list_files, read_file and search_files run straight away;
// write_file and edit_file only propose a change, which the page shows as a diff and makes only when the user clicks
// Apply (with a fingerprint of the file as the user saw it, so a file changed since isn't overwritten). Nothing
// outside the folder can be read or written, and no commands are run. The server only offers this to the page on
// the computer running tierlm (see the checks in server.cpp).
#pragma once

#include <cctype>
#include <cstdlib>
#include "agent.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace workspace {

using json = nlohmann::json;
namespace fs = std::filesystem;

struct Error : std::runtime_error { using std::runtime_error::runtime_error; };

// The tools the model is given (OpenAI format).
inline agent::ojson tools() {
    using oj = agent::ojson;
    auto fn = [](const char* name, const char* desc, oj props, oj required) {
        oj params = oj::object();
        params["type"] = "object";
        params["properties"] = props;
        params["required"] = required;
        oj f = oj::object();
        f["name"] = name; f["description"] = desc; f["parameters"] = params;
        oj t = oj::object();
        t["type"] = "function"; t["function"] = f;
        return t;
    };
    auto prop = [](const char* type, const char* desc) { oj p = oj::object(); p["type"] = type; p["description"] = desc; return p; };
    auto props = [](std::initializer_list<std::pair<const char*, oj>> l) { oj o = oj::object(); for (auto& [k, v] : l) o[k] = v; return o; };
    oj t = oj::array();
    t.push_back(fn("list_files", "List the files and folders in a folder of the workspace. Folders end with /.",
                   props({{"path", prop("string", "Folder, relative to the workspace; empty for the workspace itself")},
                          {"recursive", prop("boolean", "Also list what's in the subfolders (a few levels deep)")}}), oj::array()));
    t.push_back(fn("read_file", "Read a text file in the workspace.",
                   props({{"path", prop("string", "File path, relative to the workspace")},
                          {"start_line", prop("integer", "First line to read (1-based); optional")},
                          {"end_line", prop("integer", "Last line to read; optional")}}), oj::array({"path"})));
    t.push_back(fn("search_files", "Find the lines that contain some text (not case sensitive) in the workspace's files.",
                   props({{"query", prop("string", "The text to look for")},
                          {"path", prop("string", "Folder or file to search, relative to the workspace; optional")}}), oj::array({"query"})));
    t.push_back(fn("write_file", "Create a file, or replace a file's whole content. The user sees the change and approves it before it is made.",
                   props({{"path", prop("string", "File path, relative to the workspace")},
                          {"content", prop("string", "The complete new content of the file")}}), oj::array({"path", "content"})));
    t.push_back(fn("edit_file", "Replace one piece of text in a file. old_text must appear exactly once in the file, copied exactly, "
                                "including indentation. The user sees the change and approves it before it is made.",
                   props({{"path", prop("string", "File path, relative to the workspace")},
                          {"old_text", prop("string", "The exact text to replace")},
                          {"new_text", prop("string", "The text to put in its place")}}), oj::array({"path", "old_text", "new_text"})));
    return t;
}

// What the model is told about the workspace (added to the system prompt).
inline std::string instructions(const std::string& folder) {
    return "You are working with the files in the workspace folder " + folder + " on the user's computer. Use the tools to look "
           "at files before you answer questions about them or change them; paths are relative to the workspace. write_file and "
           "edit_file only propose a change: the user sees it and decides whether to make it. Prefer edit_file for small changes "
           "to existing files. You can't run commands.";
}

inline bool changes_files(const std::string& tool) { return tool == "write_file" || tool == "edit_file"; }

// A path from the model, as a file inside the workspace. Anything that leads outside it (.., an absolute path
// elsewhere, a link pointing out) is refused.
inline fs::path inside(const fs::path& root, const std::string& rel) {
    std::error_code ec;
    const fs::path base = fs::weakly_canonical(root, ec);
    if (ec || !fs::is_directory(base, ec)) throw Error("the workspace folder doesn't exist any more: choose it again in tierlm's settings");
    std::string r = agent::trim(rel);
    while (!r.empty() && (r == "." || r == "./" || r == ".\\")) r.clear();
    // A path with a drive, a root or a network share (\\server\share) is refused by its spelling, before the file
    // system is asked anything (looking a share up would contact that server), unless it spells out a place inside
    // the workspace.
    {
        const fs::path rp = fs::u8path(r);
        if (rp.has_root_name() || rp.has_root_directory()) {
            const fs::path in = rp.lexically_normal().lexically_relative(base);
            if (in.empty() || *in.begin() == "..")
                throw Error(rel + " is outside the workspace; only files inside it can be used (give paths relative to the workspace)");
        }
    }
    const fs::path full = fs::weakly_canonical(r.empty() ? base : base / fs::u8path(r), ec);
    if (ec) throw Error("can't use the path " + rel);
    const fs::path inner = full.lexically_relative(base);
    if (inner.empty() || *inner.begin() == "..") throw Error(rel + " is outside the workspace; only files inside it can be used");
    return full;
}

inline std::string shown(const fs::path& root, const fs::path& p) {
    std::error_code ec;
    const fs::path r = p.lexically_relative(fs::weakly_canonical(root, ec));
    return r.empty() || r == "." ? std::string(".") : r.generic_u8string();
}

inline std::string slurp(const fs::path& p, size_t cap) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw Error("can't open " + p.filename().u8string());
    std::string s(cap, '\0');
    f.read(&s[0], (std::streamsize)cap);
    s.resize((size_t)f.gcount());
    return s;
}

inline bool binary(const std::string& s) { const size_t z = s.find('\0'); return z != std::string::npos && z < 8192; }

inline bool skip_dir(const std::string& name) {
    static const char* skip[] = {".git", "node_modules", "__pycache__", ".venv", "venv", ".vs", ".idea", ".svn", ".hg"};
    for (const char* s : skip) if (name == s) return true;
    return false;
}

inline std::vector<std::string> lines_of(const std::string& s) {
    std::vector<std::string> v;
    size_t a = 0;
    while (a < s.size()) {
        size_t b = s.find('\n', a);
        if (b == std::string::npos) b = s.size();
        std::string line = s.substr(a, b - a);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        v.push_back(line);
        a = b + 1;
    }
    return v;
}

inline std::string size_text(uintmax_t sz) {
    return sz < 1024 ? std::to_string(sz) + " B" : sz < 1048576 ? std::to_string(sz / 1024) + " KB" : std::to_string(sz / 1048576) + " MB";
}

// Tool arguments as models send them. Small models often quote booleans and numbers ("recursive": "true",
// "start_line": "12"); read those as meant rather than fail.
inline std::string arg_str(const json& a, const char* k) {
    if (!a.is_object() || !a.contains(k) || a[k].is_null()) return std::string();
    return a[k].is_string() ? a[k].get<std::string>() : a[k].dump();
}
inline bool arg_bool(const json& a, const char* k, bool def) {
    if (!a.is_object() || !a.contains(k)) return def;
    const json& v = a[k];
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number()) return v.get<double>() != 0;
    if (v.is_string()) {
        std::string t = v.get<std::string>();
        for (char& ch : t) ch = (char)std::tolower((unsigned char)ch);
        if (t == "true" || t == "yes" || t == "1" || t == "on") return true;
        if (t == "false" || t == "no" || t == "0" || t == "off" || t.empty()) return false;
    }
    return def;
}
inline bool arg_int(const json& a, const char* k, int& out) {
    if (!a.is_object() || !a.contains(k)) return false;
    const json& v = a[k];
    if (v.is_number()) { out = (int)v.get<double>(); return true; }
    if (v.is_string()) {
        const std::string t = v.get<std::string>();
        char* end = nullptr;
        const long n = std::strtol(t.c_str(), &end, 10);
        if (end && end != t.c_str()) { out = (int)n; return true; }
    }
    return false;
}

inline std::string list_files(const fs::path& root, const json& a) {
    const std::string rel = arg_str(a, "path");
    const fs::path dir = inside(root, rel);
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) throw Error((rel.empty() ? "the workspace" : rel) + " is not a folder");
    const bool rec = arg_bool(a, "recursive", false);
    std::string out;
    int count = 0;
    const int cap = 400;
    std::function<void(const fs::path&, int)> walk = [&](const fs::path& d, int depth) {
        std::vector<fs::directory_entry> es;
        std::error_code e1;
        for (fs::directory_iterator it(d, fs::directory_options::skip_permission_denied, e1), end; !e1 && it != end; it.increment(e1)) es.push_back(*it);
        std::sort(es.begin(), es.end(), [](const fs::directory_entry& x, const fs::directory_entry& y) {
            std::error_code e;
            const bool dx = x.is_directory(e), dy = y.is_directory(e);
            return dx != dy ? dx : x.path().filename() < y.path().filename();
        });
        for (const auto& e : es) {
            if (count >= cap) return;
            std::error_code e2;
            const std::string name = e.path().filename().u8string();
            if (e.is_directory(e2)) {
                out += shown(root, e.path()) + "/\n";
                ++count;
                if (rec && depth < 4 && !skip_dir(name) && !e.is_symlink(e2)) walk(e.path(), depth + 1);
            } else {
                out += shown(root, e.path()) + "  (" + size_text(e.file_size(e2)) + ")\n";
                ++count;
            }
        }
    };
    walk(dir, 0);
    if (out.empty()) out = "(empty folder)\n";
    if (count >= cap) out += "(stopped after " + std::to_string(cap) + " entries)\n";
    return out;
}

inline std::string read_text(const fs::path& f, const std::string& name) {
    std::error_code ec;
    if (!fs::is_regular_file(f, ec)) throw Error(name + " doesn't exist or is not a file");
    if (fs::file_size(f, ec) > 8u * 1048576) throw Error(name + " is larger than 8 MB; only smaller text files can be read");
    std::string s = slurp(f, 8u * 1048576);
    if (binary(s)) throw Error(name + " is not a text file");
    return s;
}

inline std::string read_file(const fs::path& root, const json& a) {
    const std::string name = arg_str(a, "path");
    const std::string s = read_text(inside(root, name), name);
    const auto lines = lines_of(s);
    const int total = (int)lines.size();
    if (total == 0) return "(" + name + " is empty)\n";
    const bool ranged = a.contains("start_line") || a.contains("end_line");
    int from = 1; arg_int(a, "start_line", from);
    int to = total; arg_int(a, "end_line", to);
    from = std::max(1, std::min(from, total));
    to = std::max(from, std::min(total, to));
    std::string out;
    size_t bytes = 0;
    int last = from - 1;
    for (int i = from; i <= to && i - from < 2000 && bytes <= 200000; ++i) {
        out += lines[i - 1] + "\n";
        bytes += lines[i - 1].size() + 1;
        last = i;
    }
    if (ranged || last < total)
        out = "[" + name + ": lines " + std::to_string(from) + "-" + std::to_string(last) + " of " + std::to_string(total) + "]\n" + out +
              (last < total ? "[" + std::to_string(total - last) + " more lines: read again with start_line=" + std::to_string(last + 1) + "]\n" : "");
    return out;
}

inline std::string lower(std::string s) { for (auto& c : s) c = (char)std::tolower((unsigned char)c); return s; }

inline std::string search_files(const fs::path& root, const json& a) {
    const std::string q = lower(arg_str(a, "query"));
    if (q.empty()) throw Error("search_files needs some text to look for");
    const fs::path where = inside(root, arg_str(a, "path"));
    std::string out;
    int hits = 0, files = 0;
    auto look = [&](const fs::path& f) {
        std::error_code e;
        if (++files > 5000 || fs::file_size(f, e) > 2u * 1048576) return;
        std::string s;
        try { s = slurp(f, 2u * 1048576); } catch (...) { return; }
        if (binary(s)) return;
        const auto lines = lines_of(s);
        for (size_t i = 0; i < lines.size() && hits < 100; ++i)
            if (lower(lines[i]).find(q) != std::string::npos) {
                std::string l = agent::trim(lines[i]);
                if (l.size() > 200) l = l.substr(0, 200) + "...";
                out += shown(root, f) + ":" + std::to_string(i + 1) + ": " + l + "\n";
                ++hits;
            }
    };
    std::error_code ec;
    if (fs::is_regular_file(where, ec)) look(where);
    else {
        fs::recursive_directory_iterator it(where, fs::directory_options::skip_permission_denied, ec), end;
        for (; !ec && it != end && hits < 100; it.increment(ec)) {
            std::error_code e2;
            // (links and junctions aren't followed: one pointing outside the workspace would be read through)
            if (it->is_symlink(e2)) { it.disable_recursion_pending(); continue; }
            if (it->is_directory(e2)) {
                if (skip_dir(it->path().filename().u8string())) { it.disable_recursion_pending(); continue; }
                try { inside(root, shown(root, it->path())); } catch (...) { it.disable_recursion_pending(); }
                continue;
            }
            if (it->is_regular_file(e2)) look(it->path());
        }
    }
    if (hits == 0) return "no lines contain \"" + arg_str(a, "query") + "\"\n";
    if (hits >= 100) out += "(stopped after 100 matches)\n";
    return out;
}

// A unified diff (3 lines of context) between two versions of a file.
inline std::string unified_diff(const std::string& before, const std::string& after, const std::string& name, bool created) {
    const auto A = lines_of(before), B = lines_of(after);
    size_t pre = 0;
    while (pre < A.size() && pre < B.size() && A[pre] == B[pre]) ++pre;
    size_t suf = 0;
    while (suf < A.size() - pre && suf < B.size() - pre && A[A.size() - 1 - suf] == B[B.size() - 1 - suf]) ++suf;
    const size_t n = A.size() - pre - suf, m = B.size() - pre - suf;
    std::vector<std::pair<char, const std::string*>> ops;
    for (size_t i = 0; i < pre; ++i) ops.push_back({' ', &A[i]});
    if (n * m <= 4000000) {                                  // longest common subsequence of the changed middle
        std::vector<std::vector<int>> L(n + 1, std::vector<int>(m + 1, 0));
        for (size_t i = n; i-- > 0;)
            for (size_t j = m; j-- > 0;)
                L[i][j] = A[pre + i] == B[pre + j] ? L[i + 1][j + 1] + 1 : std::max(L[i + 1][j], L[i][j + 1]);
        size_t i = 0, j = 0;
        while (i < n || j < m) {
            if (i < n && j < m && A[pre + i] == B[pre + j]) { ops.push_back({' ', &A[pre + i]}); ++i; ++j; }
            else if (i < n && (j == m || L[i + 1][j] >= L[i][j + 1])) { ops.push_back({'-', &A[pre + i]}); ++i; }   // removed lines first
            else { ops.push_back({'+', &B[pre + j]}); ++j; }
        }
    } else {
        for (size_t i = 0; i < n; ++i) ops.push_back({'-', &A[pre + i]});
        for (size_t j = 0; j < m; ++j) ops.push_back({'+', &B[pre + j]});
    }
    for (size_t i = A.size() - suf; i < A.size(); ++i) ops.push_back({' ', &A[i]});

    std::string out = "--- " + (created ? std::string("/dev/null") : "a/" + name) + "\n+++ b/" + name + "\n";
    const size_t ctx = 3;
    size_t k = 0;
    while (k < ops.size()) {
        while (k < ops.size() && ops[k].first == ' ') ++k;
        if (k == ops.size()) break;
        const size_t start = k >= ctx ? k - ctx : 0;
        size_t end = k;
        for (;;) {                                           // join changes closer than 2*ctx unchanged lines
            while (end < ops.size() && ops[end].first != ' ') ++end;
            size_t gap = end;
            while (gap < ops.size() && ops[gap].first == ' ' && gap - end < 2 * ctx + 1) ++gap;
            if (gap < ops.size() && ops[gap].first != ' ') { end = gap; continue; }
            break;
        }
        const size_t stop = std::min(ops.size(), end + ctx);
        size_t a0 = 1, b0 = 1, na = 0, nb = 0;
        for (size_t i = 0; i < start; ++i) { if (ops[i].first != '+') ++a0; if (ops[i].first != '-') ++b0; }
        for (size_t i = start; i < stop; ++i) { if (ops[i].first != '+') ++na; if (ops[i].first != '-') ++nb; }
        out += "@@ -" + std::to_string(na ? a0 : a0 - 1) + "," + std::to_string(na) + " +" + std::to_string(nb ? b0 : b0 - 1) + "," +
               std::to_string(nb) + " @@\n";
        for (size_t i = start; i < stop; ++i) out += std::string(1, ops[i].first) + *ops[i].second + "\n";
        k = stop;
    }
    return out;
}

// A change the model asks for: the file, and its content before and after.
struct Change { fs::path file; std::string name, before, after; bool exists = false; };

inline std::string crlf(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) { if (s[i] == '\n' && (i == 0 || s[i - 1] != '\r')) o += '\r'; o += s[i]; }
    return o;
}

inline Change plan(const fs::path& root, const std::string& tool, const json& a) {
    Change c;
    c.name = arg_str(a, "path");
    if (agent::trim(c.name).empty()) throw Error(tool + " needs a path");
    c.file = inside(root, c.name);
    c.name = shown(root, c.file);                            // relative to the workspace, however the model wrote it
    std::error_code ec;
    if (fs::is_directory(c.file, ec)) throw Error(c.name + " is a folder");
    c.exists = fs::exists(c.file, ec);
    if (c.exists) c.before = read_text(c.file, c.name);
    const bool windows_lines = c.before.find("\r\n") != std::string::npos;   // keep the file's line endings
    if (tool == "write_file") {
        if (!a.contains("content") || !a["content"].is_string()) throw Error("write_file needs the file's content");
        c.after = a["content"].get<std::string>();
        if (windows_lines) c.after = crlf(c.after);
    } else if (tool == "edit_file") {
        if (!c.exists) throw Error(c.name + " doesn't exist; use write_file to create it");
        std::string from = arg_str(a, "old_text"), to = arg_str(a, "new_text");
        if (from.empty()) throw Error("edit_file needs old_text: the exact text to replace");
        if (windows_lines) { from = crlf(from); to = crlf(to); }
        const size_t at = c.before.find(from);
        size_t n = 0;
        for (size_t p = at; p != std::string::npos; p = c.before.find(from, p + 1)) ++n;
        if (n == 0) throw Error("old_text was not found in " + c.name + ": read the file again and copy the text exactly");
        if (n > 1) throw Error("old_text appears " + std::to_string(n) + " times in " + c.name + ": include more of the surrounding lines so it appears once");
        c.after = c.before.substr(0, at) + to + c.before.substr(at + from.size());
    } else throw Error("unknown change " + tool);
    return c;
}

// A short fingerprint of a file's content, so a change is only made to the file the user saw the diff of.
inline std::string fingerprint(bool exists, const std::string& s) {
    uint64_t h = 1469598103934665603ull;                     // FNV-1a
    for (unsigned char ch : s) { h ^= ch; h *= 1099511628211ull; }
    char buf[24];
    std::snprintf(buf, sizeof buf, "%c%016llx", exists ? 'f' : 'n', (unsigned long long)h);
    return buf;
}

inline void apply(const Change& c) {
    std::error_code ec;
    if (c.file.has_parent_path()) fs::create_directories(c.file.parent_path(), ec);
    std::ofstream f(c.file, std::ios::binary | std::ios::trunc);
    if (!f) throw Error("can't write " + c.name);
    f.write(c.after.data(), (std::streamsize)c.after.size());
    if (!f) throw Error("writing " + c.name + " failed");
}

// Runs one tool call. apply = false for a change: returns its diff and fingerprint (nothing is written); apply = true:
// makes it, if the file still has that fingerprint. Returns {"result"} for the model, plus {"diff", "hash"} for changes.
inline json run(const fs::path& root, const std::string& tool, const json& args, bool do_apply, const std::string& hash) {
    if (tool == "list_files") return {{"result", list_files(root, args)}};
    if (tool == "read_file") return {{"result", read_file(root, args)}};
    if (tool == "search_files") return {{"result", search_files(root, args)}};
    if (!changes_files(tool)) throw Error("there is no tool called " + tool);
    const Change c = plan(root, tool, args);
    const std::string fp = fingerprint(c.exists, c.before);
    json r = {{"diff", unified_diff(c.before, c.after, c.name, !c.exists)}, {"hash", fp}, {"path", c.name}, {"created", !c.exists}};
    if (!do_apply) {
        // The whole file before and after, for the page's side-by-side view (not for very big files).
        if (c.before.size() + c.after.size() <= 4u * 1048576) { r["before"] = c.before; r["after"] = c.after; }
        return r;
    }
    if (hash != fp) throw Error(c.name + " changed since the change was shown; it was not made");
    apply(c);
    r["result"] = (c.exists ? "Changed " : "Created ") + c.name + ".";
    return r;
}

}  // namespace workspace
