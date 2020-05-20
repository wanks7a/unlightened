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

Uint32 leprGrayScale(float input, float max)
{
    if (input < 0)
        input = 0;

    Uint32 value = (input / max) * 255;
    Uint32 result = 0;
    result |= value;
    result |= (value << 8);
    result |= (value << 16);
    return result;
}

void shape_plot::draw()
{
    {
        std::lock_guard<std::mutex> lock(m);
        if (!new_data)
            return;
        else
            new_data = false;
    }
    
    std::vector<SDL_Surface*> surfaces;
    std::vector<SDL_Texture*> images;
    std::vector<SDL_Rect> screen_dest;

    std::vector<Uint32> single_pic(sh.area());
    Uint32 rmask, gmask, bmask, amask;
    rmask = 0x000000ff;
    gmask = 0x0000ff00;
    bmask = 0x00ff0000;
    amask = 0;
    int startW = 0, startH = 0;
    for (size_t i = 0; i < sh.depth; i++)
    {
        float max = findMax(&data[i * sh.area()], sh.area());
        for (size_t j = 0; j < sh.area(); j++)
        {
            single_pic[j] = leprGrayScale(data[i * sh.area() + j], max);
        }
        SDL_Surface* surf = SDL_CreateRGBSurfaceFrom((void*)single_pic.data(), sh.width, sh.height, 32, 4 * sh.width,
            rmask, gmask, bmask, amask);
        SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surf);
        images.push_back(texture);
        surfaces.push_back(surf);
        SDL_Rect dstrect = { startW, startH, sh.width * 4, sh.height * 4 };
        screen_dest.push_back(dstrect);
        if ((startW + sh.width * 4) >= options.w)
        {
            startW = 0;
            startH += sh.height * 4;
        }
        else
        {
            startW += sh.width * 4;
        }
    }

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

void shape_plot::set_shape_data(std::vector<float>&& input, const shape& sh)
{
	std::lock_guard<std::mutex> lock(m);
	data = std::move(input);
	this->sh = sh;
    new_data = true;
}