#include <shape_plot.h>

float findMax(const float* ptr, size_t size)
{
    float max = ptr[0];
    for (size_t i = 0; i < size; i++)
    {
        if (max < ptr[i])
            max = ptr[i];
    }
    return max;
}

Uint32 GrayScale(float input)
{
    Uint32 result = input;
    result |= result;
    result |= (result << 8);
    result |= (result << 16);
    return result;
}

void shape_plot::draw()
{
    if (new_data)
    {
        std::lock_guard<std::mutex> lock(m);
        rgb_data = std::move(swap_data);
        new_data = false;
    }

    if(latest_draw_call)
        latest_draw_call(*this);
}

shape_plot::shape_plot(int w, int h, std::string name) : view(w, h, name)
{
}

bool shape_plot::draw_rgb_channels(std::vector<float>&& rgb_image, size_t width, size_t height, float min_val, float max_val)
{
    if(rgb_image.size() == width * height * 3)
    {
        std::lock_guard<std::mutex> lock(m);
        swap_data.data = std::move(rgb_image);
        swap_data.width = width;
        swap_data.height = height;
        swap_data.min_val = min_val;
        swap_data.max_val = max_val;
        latest_draw_call = &shape_plot::draw_rgb_channels_internal;
        new_data = true;
        return true;
    }
    return false;
}

void shape_plot::draw_rgb_channels_internal()
{
    std::vector<SDL_Surface*> surfaces;
    std::vector<SDL_Texture*> images;
    std::vector<SDL_Rect> screen_dest;

    std::vector<Uint32> single_pic(rgb_data.width * rgb_data.height);

    Uint32 rmask = 0x000000ff;
    Uint32 gmask = 0x0000ff00;
    Uint32 bmask = 0x00ff0000;
    Uint32 amask = 0;
    int startW = 0, startH = 0;

    for (size_t j = 0; j < single_pic.size(); j++)
    {
        single_pic[j] = 0;
        float value = map_val(rgb_data.data[j], rgb_data.min_val, rgb_data.max_val, 0.0f, 255.0f);
        single_pic[j] |= static_cast<unsigned char>(value); // Red
        value = map_val(rgb_data.data[j + rgb_data.width * rgb_data.height], rgb_data.min_val, rgb_data.max_val, 0.0f, 255.0f);
        single_pic[j] |= static_cast<Uint32>(static_cast<unsigned char>(value)) << 8; // Red
        value = map_val(rgb_data.data[j + 2 * rgb_data.width * rgb_data.height], rgb_data.min_val, rgb_data.max_val, 0.0f, 255.0f);
        single_pic[j] |= static_cast<Uint32>(static_cast<unsigned char>(value)) << 16; // Red
    }

    SDL_Surface* surf = SDL_CreateRGBSurfaceFrom((void*)single_pic.data(), rgb_data.width, rgb_data.height, 32, 4 * rgb_data.width,
        rmask, gmask, bmask, amask);
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surf);
    images.push_back(texture);
    surfaces.push_back(surf);
    SDL_Rect dstrect = { startW, startH, static_cast<int>(rgb_data.width * scale), static_cast<int>(rgb_data.height * scale) };
    screen_dest.push_back(dstrect);
    
    for (size_t i = 0; i < images.size(); i++)
    {
        SDL_RenderCopy(renderer, images[i], NULL, &screen_dest[i]);
    }

    SDL_RenderPresent(renderer);
    for (size_t i = 0; i < images.size(); i++)
    {
        SDL_DestroyTexture(images[i]);
        SDL_FreeSurface(surfaces[i]);
    }
}

bool shape_plot::draw_grayscale(std::vector<float>&& image, size_t width, size_t height, float min_val, float max_val)
{
    if (image.size() == width * height)
    {
        std::lock_guard<std::mutex> lock(m);
        swap_data.data = std::move(image);
        swap_data.width = width;
        swap_data.height = height;
        swap_data.min_val = min_val;
        swap_data.max_val = max_val;
        latest_draw_call = &shape_plot::draw_grayscale_internal;
        new_data = true;
        return true;
    }
    return false;
}

void shape_plot::draw_grayscale_internal()
{
    std::vector<SDL_Surface*> surfaces;
    std::vector<SDL_Texture*> images;
    std::vector<SDL_Rect> screen_dest;

    std::vector<Uint32> single_pic(rgb_data.width * rgb_data.height);

    Uint32 rmask = 0x000000ff;
    Uint32 gmask = 0x0000ff00;
    Uint32 bmask = 0x00ff0000;
    Uint32 amask = 0;
    int startW = 0, startH = 0;

    for (size_t j = 0; j < single_pic.size(); j++)
    {
        single_pic[j] = GrayScale(map_val(rgb_data.data[j], rgb_data.min_val, rgb_data.max_val, 0.0f, 255.0f));
    }

    SDL_Surface* surf = SDL_CreateRGBSurfaceFrom((void*)single_pic.data(), rgb_data.width, rgb_data.height, 32, 4 * rgb_data.width,
        rmask, gmask, bmask, amask);
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surf);
    images.push_back(texture);
    surfaces.push_back(surf);
    SDL_Rect dstrect = { startW, startH, static_cast<int>(rgb_data.width * scale), static_cast<int>(rgb_data.height * scale) };
    screen_dest.push_back(dstrect);

    for (size_t i = 0; i < images.size(); i++)
    {
        SDL_RenderCopy(renderer, images[i], NULL, &screen_dest[i]);
    }

    SDL_RenderPresent(renderer);
    for (size_t i = 0; i < images.size(); i++)
    {
        SDL_DestroyTexture(images[i]);
        SDL_FreeSurface(surfaces[i]);
    }
}

void shape_plot::set_scale(float val)
{
    scale = val;
}