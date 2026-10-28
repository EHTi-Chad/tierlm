#pragma once
#include <string>
#include <vector>

namespace tierlm {

// Big models are often published as several GGUF files ("split" GGUFs):
//   model-00001-of-00003.gguf   settings, tokenizer and some weights
//   model-00002-of-00003.gguf   weights only
//   model-00003-of-00003.gguf   weights only
// Each part carries split.no / split.count. Part 1 is the one to open.
struct SplitInfo {
    int count = 1;                      // number of parts (1 = an ordinary single file)
    std::vector<std::string> paths;     // every part, in order
    std::string missing;                // file name of the first missing part, if any
};

// Reads part 1's header and finds the other parts next to it. Throws if the file can't be read.
SplitInfo gguf_split_info(const std::string& path);

// True for part 2, 3, ... of a split model (by file name), which should not be listed as models of their own.
bool gguf_is_later_part(const std::string& filename);

}
