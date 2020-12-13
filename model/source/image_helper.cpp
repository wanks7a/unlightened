#include <image_helper.h>

image_info load_image_normalized(const char* path)
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

image_info load_image_normalized(const char* path, size_t max_width, size_t max_height)
{
    image_info result;
    SDL_Surface* surf = IMG_Load(path);

    int width = surf->w;
    int height = surf->h;

    if ((width > max_width) ||
        (height > max_height)) {
        SDL_Rect sourceDimensions;
        sourceDimensions.x = 0;
        sourceDimensions.y = 0;
        sourceDimensions.w = width;
        sourceDimensions.h = height;

        float scale = (float)max_width / (float)width;
        float scaleH = (float)max_height / (float)height;

        if (scaleH < scale) {
            scale = scaleH;
        }

        SDL_Rect targetDimensions;
        targetDimensions.x = 0;
        targetDimensions.y = 0;
        targetDimensions.w = (int)(width * scale);
        targetDimensions.h = (int)(height * scale);

        SDL_Surface* pScaleSurface = SDL_CreateRGBSurface(
            surf->flags,
            targetDimensions.w,
            targetDimensions.h,
            surf->format->BitsPerPixel,
            surf->format->Rmask,
            surf->format->Gmask,
            surf->format->Bmask,
            surf->format->Amask);

        if (SDL_BlitScaled(surf, NULL, pScaleSurface, &targetDimensions) < 0) {
            printf("Error did not scale surface: %s\n", SDL_GetError());

            SDL_FreeSurface(pScaleSurface);
            pScaleSurface = NULL;
        }
        else {
            SDL_FreeSurface(surf);

            surf = pScaleSurface;
            width = surf->w;
            height = surf->h;
        }
    }

    size_t pixels = static_cast<size_t>(surf->w) * surf->h;
    result.pixels.resize(max_width * max_height * 3, 0.0f);

    for (size_t i = 0; i < pixels; i++)
    {
        Uint8 r, g, b;
        SDL_GetRGB(static_cast<Uint32>(i), surf->format, &r, &g, &b);
        result.pixels[i] = r / 255.0f;
        result.pixels[i + max_width * max_height] = g / 255.0f;
        result.pixels[i + 2 * max_width * max_height] = b / 255.0f;
    }
    result.w = max_width;
    result.h = max_height;

    SDL_FreeSurface(surf);

    return result;
}