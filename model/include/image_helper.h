#pragma once
#include <vector>
#include <SDL_image.h>

struct image_info
{
	size_t w;
	size_t h;
	std::vector<float> pixels;
};
image_info load_image_rgb(const char* path);