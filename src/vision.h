#pragma once
// Image reading: the vision encoder of Qwen3.5 (and Qwen3-VL-style models without DeepStack), read from the
// model's "mmproj" GGUF file. It turns an image into "image words": vectors the chat model reads in place of
// token embeddings, one per 32x32 pixels (16x16 patches, merged 2x2).
//
// The steps, as Qwen's reference code does them:
//   1. resize: the image is scaled (keeping its shape) so both sides are multiples of 32 and the pixel count is
//      within the limits, then normalised per colour channel;
//   2. patches: cut into 16x16 squares, listed 2x2 block by 2x2 block (the order the merge step needs);
//   3. a linear layer turns each patch into a vector; a learned position table (48x48, bilinearly resampled to
//      the image's grid) is added;
//   4. transformer blocks (LayerNorm, attention over the whole image with 2D rotary positions, GELU MLP);
//   5. merge: LayerNorm, then each 2x2 block of patches becomes one vector and an MLP maps it to the chat model's
//      width.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend;
struct ggml_backend_sched;
struct ggml_backend_buffer;
struct ggml_context;
struct ggml_tensor;
struct gguf_context;

namespace tierlm {

// A decoded image: 8-bit RGB, rows top to bottom.
struct Image {
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;                    // width * height * 3
};

// PNG, JPEG, BMP or GIF (first frame) bytes to RGB (WebP not yet). Throws if the bytes aren't an image it can read.
Image decode_image(const uint8_t* data, size_t size);

// An image's width and height from its header (without decoding it); false if it isn't one stb_image reads.
bool image_size(const uint8_t* data, size_t size, int* w, int* h);

// True if the file is an mmproj (vision) GGUF this engine can read.
bool is_vision_file(const std::string& path, std::string* why = nullptr);

class Vision {
public:
    // device: "auto" (a GPU with room to spare, else the CPU), "cpu" or a ggml device name. Throws on failure.
    static std::unique_ptr<Vision> load(const std::string& path, const std::string& device = "auto");
    ~Vision();

    // An image made ready for encode(): resized and cut into patches. max_tokens limits the image words it
    // becomes (bigger images are scaled down); 0: the model's own limit.
    struct Prepared {
        int gh = 0, gw = 0;                      // grid of 16x16 patches (both even)
        int width = 0, height = 0;               // the resized image
        std::vector<float> patches;              // gh*gw patches of 3*16*16 numbers, in merge-block order
        int tokens() const { return gh * gw / 4; }
    };
    Prepared prepare(const Image& img, int max_tokens = 0) const;
    // The image words a width x height image becomes (what prepare() would give), without the image.
    int tokens_for(int width, int height, int max_tokens = 0) const;
    // The image words: tokens() vectors of out_dim() numbers each.
    std::vector<float> encode(const Prepared& p, int n_threads);

    int out_dim() const { return out_dim_; }
    int merge() const { return merge_; }
    int patch() const { return patch_; }
    int min_tokens() const { return min_tokens_; }
    int max_tokens() const { return max_tokens_; }
    const std::string& device_name() const { return device_name_; }
    size_t weight_bytes() const { return weight_bytes_; }
    bool on_gpu() const { return main_ != cpu_; }
    bool gpu_helps() const { return help_ != nullptr; }   // weights in RAM, a GPU doing the heavy work
    bool last_helped() const { return last_helped_; }      // the last image: did it (or was there no room)

private:
    Vision() = default;
    void resized(int w, int h, int max_tokens, int& rw, int& rh) const;   // Qwen's smart resize
    struct Block {
        ggml_tensor *ln1 = nullptr, *ln1_b = nullptr, *qkv = nullptr, *qkv_b = nullptr, *out = nullptr, *out_b = nullptr;
        ggml_tensor *ln2 = nullptr, *ln2_b = nullptr, *up = nullptr, *up_b = nullptr, *down = nullptr, *down_b = nullptr;
    };
    std::string device_name_;
    int n_embd_ = 0, n_head_ = 0, n_ff_ = 0, patch_ = 16, merge_ = 2, out_dim_ = 0, pos_side_ = 0;
    int min_tokens_ = 4, max_tokens_ = 4096;
    float eps_ = 1e-6f, rope_base_ = 10000.0f, mean_[3] = {0.5f, 0.5f, 0.5f}, std_[3] = {0.5f, 0.5f, 0.5f};
    ggml_tensor *patch_w_ = nullptr, *patch_b_ = nullptr, *pos_ = nullptr, *post_ln_ = nullptr, *post_ln_b_ = nullptr;
    ggml_tensor *mm0_ = nullptr, *mm0_b_ = nullptr, *mm2_ = nullptr, *mm2_b_ = nullptr;
    std::vector<Block> blocks_;
    ggml_context* wctx_ = nullptr;               // the file's tensors
    ggml_context* xctx_ = nullptr;               // tensors tierlm makes (the patch weights as one 2D matrix)
    gguf_context* gguf_ = nullptr;
    ggml_backend* cpu_ = nullptr;
    ggml_backend* main_ = nullptr;
    ggml_backend_buffer* wbuf_ = nullptr;
    ggml_backend_buffer* xbuf_ = nullptr;
    ggml_backend_sched* sched_ = nullptr;
    ggml_backend* help_ = nullptr;               // weights in RAM: a GPU that still does the heavy work
    ggml_backend_sched* sched_help_ = nullptr;
    bool last_helped_ = false;
    size_t weight_bytes_ = 0;
};

}
