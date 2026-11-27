// The link between RAM and a GPU, measured (0.9.7): how fast data goes to the card and back over PCIe (or
// whatever joins them), the way tierlm moves it. Works through ggml, so on CUDA and Vulkan alike.
#pragma once
#include <string>

namespace tierlm {

struct LinkTest {
    bool ok = false;
    std::string error;                 // (not ok: why)
    std::string device;                // the GPU measured
    size_t big = 0;                    // bytes of the big copies
    bool pinned = false;               // pinned memory was available (else the pinned numbers are 0)
    double to_card = 0;                // GB/s: big copies RAM -> card, pinned memory (how long prompts' experts go)
    double to_ram = 0;                 // GB/s: big copies card -> RAM, pinned memory
    double to_card_plain = 0;          // GB/s: big copies RAM -> card, ordinary memory
    double pieces_each = 0;            // GB/s: 2 MB pieces RAM -> card, pinned, each waited for
    double pieces_queued = 0;          // GB/s: the same, queued and waited for once (how experts are moved into VRAM)
    double seconds = 0;                // how long the test took
};

// Measures the GPU whose name or description contains `device` (empty: the first GPU). Uses up to 512 MB of the
// card's memory and twice that of RAM for a few seconds; nothing else should be using the card meanwhile.
LinkTest link_test(const std::string& device);

} // namespace tierlm
