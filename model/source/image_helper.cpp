#include <image_helper.h>

image_info load_image_rgb(const char* path)
{
    image_info result;
    SDL_Surface* surf = IMG_Load(path);
    size_t pixels = static_cast<size_t>(surf->w )* surf->h;
    result.pixels.resize(pixels * 3);
    for (size_t i = 0; i < pixels; i++)
    {
        Uint8 r, g, b;
        SDL_GetRGB(static_cast<Uint32*>(surf->pixels)[i], surf->format, &r, &g, &b);
        result.pixels[i] = r / 255.0f;
        result.pixels[i + pixels] = g / 255.0f;
        result.pixels[i + 2 * pixels] = b / 255.0f;
    }
    result.w = surf->w;
    result.h = surf->h;
    SDL_FreeSurface(surf);
    return result;
}