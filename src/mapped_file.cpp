#include "mapped_file.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT 0x0A00             // Windows 10: PrefetchVirtualMemory
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

#ifdef _WIN32
static std::wstring widen(const std::string& s) {          // UTF-8 path to Windows' UTF-16
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
    return w;
}
#endif

void MappedFile::open(const std::string& path) {
    path_ = path;
#ifdef _WIN32
    HANDLE f = CreateFileW(widen(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path + " for the SSD tier");
    LARGE_INTEGER sz{};
    GetFileSizeEx(f, &sz);
    HANDLE mp = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mp) { CloseHandle(f); throw std::runtime_error("cannot map " + path + " for the SSD tier"); }
    void* v = MapViewOfFile(mp, FILE_MAP_READ, 0, 0, 0);
    if (!v) { CloseHandle(mp); CloseHandle(f); throw std::runtime_error("cannot map " + path + " for the SSD tier (not enough address space?)"); }
    file_ = f; mapping_ = mp; data_ = (const char*)v; size_ = (size_t)sz.QuadPart;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path + " for the SSD tier");
    struct stat st{};
    fstat(fd, &st);
    void* v = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (v == MAP_FAILED) { ::close(fd); throw std::runtime_error("cannot map " + path + " for the SSD tier"); }
    // Experts are picked in no particular order: read just what is touched, not the next megabytes as well.
    madvise(v, (size_t)st.st_size, MADV_RANDOM);
    fd_ = fd; data_ = (const char*)v; size_ = (size_t)st.st_size;
#endif
}

MappedFile::~MappedFile() {
#ifdef _WIN32
    if (data_) UnmapViewOfFile(data_);
    if (mapping_) CloseHandle((HANDLE)mapping_);
    if (file_) CloseHandle((HANDLE)file_);
#else
    if (data_) munmap((void*)data_, size_);
    if (fd_ >= 0) ::close(fd_);
#endif
}

void MappedFile::prefetch(const void* p, size_t n) {
    if (!p || !n) return;
#ifdef _WIN32
    WIN32_MEMORY_RANGE_ENTRY r{ (PVOID)p, n };
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &r, 0);
#else
    const size_t pg = (size_t)sysconf(_SC_PAGESIZE);
    const uintptr_t a = (uintptr_t)p & ~(uintptr_t)(pg - 1);
    madvise((void*)a, n + ((uintptr_t)p - a), MADV_WILLNEED);
#endif
}

void MappedFile::prefetch_many(const std::vector<std::pair<const void*, size_t>>& ranges) {
    if (ranges.empty()) return;
#ifdef _WIN32
    std::vector<WIN32_MEMORY_RANGE_ENTRY> r;
    r.reserve(ranges.size());
    for (const auto& [p, n] : ranges) if (p && n) r.push_back({ (PVOID)p, n });
    if (!r.empty()) PrefetchVirtualMemory(GetCurrentProcess(), r.size(), r.data(), 0);
#else
    for (const auto& [p, n] : ranges) prefetch(p, n);
#endif
}

std::vector<DriveInfo> list_drives() {
    std::vector<DriveInfo> out;
#ifdef _WIN32
    wchar_t buf[512];
    DWORD n = GetLogicalDriveStringsW(511, buf);
    for (const wchar_t* p = buf; n && *p; p += wcslen(p) + 1) {
        if (GetDriveTypeW(p) != DRIVE_FIXED) continue;            // not CD drives, network shares or USB sticks
        ULARGE_INTEGER fr{}, tot{};
        if (!GetDiskFreeSpaceExW(p, &fr, &tot, nullptr)) continue;
        wchar_t name[MAX_PATH + 1] = L"";
        GetVolumeInformationW(p, name, MAX_PATH, nullptr, nullptr, nullptr, nullptr, 0);
        auto narrow = [](const wchar_t* w) {
            int k = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
            std::string s(k > 0 ? k - 1 : 0, '\0');
            if (k > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), k, nullptr, nullptr);
            return s;
        };
        out.push_back({narrow(p), narrow(name), fr.QuadPart, tot.QuadPart});
    }
#else
    FILE* f = std::fopen("/proc/mounts", "r");
    if (f) {
        char dev[512], mnt[512], type[64];
        while (std::fscanf(f, "%511s %511s %63s %*[^\n]", dev, mnt, type) == 3) {
            const std::string d = dev, t = type;
            if (d.compare(0, 5, "/dev/") != 0 || t == "squashfs" || t == "iso9660") continue;   // real disks only
            if (std::string(mnt).compare(0, 5, "/boot") == 0 || std::string(mnt).compare(0, 5, "/snap") == 0) continue;
            unsigned long long fr = 0, tot = 0;
            if (!drive_space(mnt, fr, tot)) continue;
            bool dup = false;
            for (const auto& x : out) if (x.label == d) dup = true;   // the same disk mounted twice
            if (!dup) out.push_back({mnt, d, fr, tot});
        }
        std::fclose(f);
    }
#endif
    return out;
}

bool drive_space(const std::string& path, unsigned long long& free_bytes, unsigned long long& total_bytes) {
    std::error_code ec;
    auto s = std::filesystem::space(std::filesystem::u8path(path), ec);
    if (ec) return false;
    free_bytes = s.available; total_bytes = s.capacity;
    return true;
}

bool flush_to_disk(const std::string& path) {
#ifdef _WIN32
    HANDLE h = CreateFileW(widen(path).c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    const bool ok = FlushFileBuffers(h) != 0;
    CloseHandle(h);
    return ok;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
#endif
}

#ifdef _WIN32
static size_t g_kept = 0;
static bool g_have_base = false;
static SIZE_T g_base_min = 0, g_base_max = 0;
static void set_kept(size_t kept) {
    HANDLE proc = GetCurrentProcess();
    if (kept == 0) { SetProcessWorkingSetSizeEx(proc, g_base_min, g_base_max, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE); return; }
    const SIZE_T mn = g_base_min + kept + ((SIZE_T)256 << 20);
    const SIZE_T mx = (std::max)(g_base_max, mn + ((SIZE_T)1 << 30));
    SetProcessWorkingSetSizeEx(proc, mn, mx, QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
}
#endif

size_t keep_in_ram(const std::vector<std::pair<const void*, size_t>>& ranges) {
    if (const char* e = std::getenv("TIERLM_KEEP_RAM"); e && !std::strcmp(e, "0")) return 0;
    size_t total = 0;
    for (const auto& r : ranges) total += r.second;
    if (!total) return 0;
    size_t kept = 0;
#ifdef _WIN32
    HANDLE proc = GetCurrentProcess();
    if (!g_have_base) {
        DWORD flags = 0;
        if (!GetProcessWorkingSetSizeEx(proc, &g_base_min, &g_base_max, &flags)) return 0;
        g_have_base = true;
    }
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    // only with room to spare: what's kept, plus 4 GB or an eighth of RAM for everything else
    if (GlobalMemoryStatusEx(&ms) && g_kept + total + (std::max)((unsigned long long)4 << 30, ms.ullTotalPhys / 8) < ms.ullTotalPhys) {
        const SIZE_T mn = g_base_min + g_kept + total + ((SIZE_T)256 << 20);
        const SIZE_T mx = (std::max)(g_base_max, mn + ((SIZE_T)1 << 30));
        if (SetProcessWorkingSetSizeEx(proc, mn, mx, QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
            g_kept += total;
            kept = total;
        } else {
            std::fprintf(stderr, "note: couldn't reserve RAM for the weights (error %lu); they may be slow the first time they're used\n", GetLastError());
        }
    }
#endif
    // read every page once (anything already pushed out comes back now, not when a word needs it)
    volatile unsigned char sink = 0;
    for (const auto& [p, n] : ranges) {
        const unsigned char* b = (const unsigned char*)p;
        for (size_t i = 0; i < n; i += 4096) sink ^= b[i];
    }
    (void)sink;
    return kept;
}

void release_kept_ram(size_t bytes) {
#ifdef _WIN32
    if (!bytes || !g_have_base) return;
    g_kept -= (std::min)(g_kept, bytes);
    set_kept(g_kept);
#else
    (void)bytes;
#endif
}
