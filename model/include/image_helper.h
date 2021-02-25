#pragma once
#include <vector>
#include <SDL_image.h>

struct image_info
{
	size_t w;
	size_t h;
	std::vector<float> pixels;
};

struct shift_data
{
    float a;
    float b;
    constexpr shift_data(float a = 0.0f, float b = 1.0f) : a(a), b(b) {}

    float operator()(float val) const
    {
        return (val + a) / b;
    }
};

template <typename Func = shift_data>
image_info load_image(const char* path, const Func& f = Func())
{
    image_info result;
    int w, h, n;
    unsigned char* data = stbi_load(path, &w, &h, &n, STBI_rgb);
    result.pixels.reserve(w * h * 3);
    for (int i = 0; i < w * h * 3; i++)
    {
        result.pixels.emplace_back(f(data[i]));
    }
    result.w = w;
    result.h = h;
    return result;
}

image_info load_image_normalized(const char* path, size_t max_width, size_t max_height);
image_info resize_image(const image_info& img, int w, int h);
bool save_image(const char* p, const image_info& img, float pixel_scale = 1.0f);
image_info fit_image(const image_info& img, int w, int h);