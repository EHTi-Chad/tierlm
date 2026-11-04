// Decoding images for the vision encoder. PNG, JPEG, BMP, GIF (first frame), TGA, PSD, PNM and HDR are read
// by stb_image (Sean Barrett, public domain / MIT; third_party/stb). Every image becomes 8-bit RGB: an alpha
// channel is blended onto white (transparent PNGs of documents are black text on nothing).
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_FAILURE_USERMSG
#include "../third_party/stb/stb_image.h"
#include "vision.h"
#include <stdexcept>
#include <string>

namespace tierlm {

bool image_size(const uint8_t* data, size_t size, int* w, int* h) {
    int ch = 0;
    return data && size > 0 && size < ((size_t)1 << 30) && stbi_info_from_memory(data, (int)size, w, h, &ch) != 0;
}

Image decode_image(const uint8_t* data, size_t size) {
    if (!data || size == 0) throw std::runtime_error("empty image");
    if (size > (size_t)1 << 30) throw std::runtime_error("image file too large");
    if (size >= 12 && std::string((const char*)data, 4) == "RIFF" && std::string((const char*)data + 8, 4) == "WEBP")
        throw std::runtime_error("WebP images aren't supported yet: send PNG or JPEG");
    int w = 0, h = 0, ch = 0;
    if (!stbi_info_from_memory(data, (int)size, &w, &h, &ch)) throw std::runtime_error(std::string("can't read the image: ") + stbi_failure_reason());
    if ((long long)w * h > 100LL * 1000 * 1000) throw std::runtime_error("image too large (more than 100 megapixels)");
    unsigned char* px = stbi_load_from_memory(data, (int)size, &w, &h, &ch, 4);
    if (!px) throw std::runtime_error(std::string("can't read the image: ") + stbi_failure_reason());
    Image img;
    img.width = w; img.height = h;
    img.rgb.resize((size_t)w * h * 3);
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        const unsigned a = px[i * 4 + 3];
        for (int c = 0; c < 3; ++c) img.rgb[i * 3 + c] = (uint8_t)((px[i * 4 + c] * a + 255 * (255 - a) + 127) / 255);
    }
    stbi_image_free(px);
    return img;
}

}
