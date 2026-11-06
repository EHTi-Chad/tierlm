#pragma once
// A file mapped into memory, read-only: the SSD tier. Weights placed here are not read at load; the operating system
// reads the parts the model touches from the drive when it touches them (an expert when a token picks it) and keeps
// recently read parts in spare RAM, so busy experts are read once. Nothing is copied into tierlm's own memory.
#include <cstddef>
#include <string>
#include <vector>

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    // Maps the whole file; throws std::runtime_error with the reason on failure.
    void open(const std::string& path);
    const char* data() const { return data_; }
    size_t size() const { return size_; }
    const std::string& path() const { return path_; }
    // Asks the OS to start reading [p, p+n) now, without waiting (a hint; does nothing where unsupported).
    static void prefetch(const void* p, size_t n);
private:
    const char* data_ = nullptr;
    size_t size_ = 0;
    std::string path_;
#ifdef _WIN32
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// The drives the SSD tier can use: fixed drives on Windows (C:\, D:\, ...), mounted disk file systems elsewhere.
struct DriveInfo { std::string path; std::string label; unsigned long long free_bytes = 0, total_bytes = 0; };
std::vector<DriveInfo> list_drives();

// Free space and total size of the drive holding `path` (a folder or file), in bytes; false if unknown.
bool drive_space(const std::string& path, unsigned long long& free_bytes, unsigned long long& total_bytes);
