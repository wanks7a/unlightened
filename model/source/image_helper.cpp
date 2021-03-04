#include <image_helper.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_resize.h>
#include <stb_image.h>
#include <stb_image_write.h>

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

image_info resize_image(const image_info& img, int w, int h)
{
    image_info result;
    result.w = w;
    result.h = h;
    result.pixels.resize(w * h * 3);
    bool success = static_cast<bool>(stbir_resize_float(img.pixels.data(), img.w, img.h, 0, result.pixels.data(), w, h, 0, 3));
    if (!success)
    {
        result.w = 0;
        result.h = 0;
        result.pixels.clear();
    }
    return result;
}

bool save_image(const char* p, const image_info& img, float pixel_scale)
{
    std::vector<unsigned char> data;
    for (size_t i = 0; i < img.pixels.size(); i++)
    {
        float pix = roundf(img.pixels[i] * pixel_scale);
        if (pix > 255.0f)
        {
            pix = 255.0f;
        }
        if (pix < 0.0f)
        {
            pix = 0.0f;
        }
        data.emplace_back(static_cast<unsigned char>(pix));
    }
    return stbi_write_png(p, img.w, img.h, 3, data.data(), img.w * 3 * sizeof(unsigned char));
}

image_info fit_image(const image_info& img, int w, int h)
{
    float scale = 1.0f;
    if (img.w > w)
    {
        scale = w / static_cast<float>(img.w);
    }
    if (img.h > h)
    {
        if (scale > (h / img.h))
            scale = h / static_cast<float>(img.h);
    }
    image_info result; 
    result.w = w;
    result.h = h;
    result.pixels.resize(result.w * result.h * 3);
    image_info resized_img = resize_image(img, img.w * scale, img.h * scale);
    size_t x_offset, y_offset;
    size_t stride = result.w * 3;
    x_offset = (result.w - resized_img.w) / 2;
    y_offset = (result.h - resized_img.h) / 2;
    for (size_t i = 0; i < resized_img.h; i++)
    {
        memcpy(&result.pixels[stride * (y_offset + i) + x_offset * 3], &resized_img.pixels[resized_img.w * 3 * i], resized_img.w * 3 * sizeof(float));
    }
    return result;
}