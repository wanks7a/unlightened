#include <loss_plot.h>

void loss_plot::add_point(float point, const std::string& name, const std::string color)
{
    std::lock_guard<std::mutex> l(m);
    lines[name].points.emplace_back(point);
    lines[name].color = color;
    if (point > max_val)
        max_val = point;
    new_data = true;
}

void loss_plot::add_points(const std::vector<float>& points, const std::string& name, const std::string color)
{
    std::lock_guard<std::mutex> l(m);
    for (size_t i = 0; i < points.size(); i++)
    {
        if (points[i] > max_val)
            max_val = points[i];
        lines[name].points.emplace_back(points[i]);
    }
    lines[name].color = color;
    new_data = true;
}

void loss_plot::draw()
{
    if (!new_data)
        return;

    mglGraph plot(0, options.w, options.h);
    plot.SetRanges(0.0, lines.begin()->second.points.size(), 0.0, max_val);
    bool will_draw = false;
    for (auto it = lines.begin(); it != lines.end(); it++)
    {
        if (it->second.points.size() > 1)
        {
            std::lock_guard<std::mutex> l(m);
            will_draw = true;
            mglData data(it->second.points);
            data.Smooth("x");
            plot.Plot(data, it->second.color.c_str());
            plot.AddLegend(it->first.c_str(), it->second.color.c_str());
        }
    }
    if (!will_draw)
        return;
    plot.Axis();
    plot.Legend();
    plot.Box();
    SDL_Surface* surf = SDL_CreateRGBSurfaceFrom((void*)plot.GetRGBA(), options.w, options.h, 32, 4 * options.w,
        0x00ff0000, 0x0000ff00, 0x000000ff, 0);
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surf);
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    view::draw();
    SDL_DestroyTexture(texture);
    SDL_FreeSurface(surf);
}

loss_plot::loss_plot()
{
}