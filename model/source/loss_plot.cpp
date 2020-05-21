#include <loss_plot.h>
#include <mgl2/mgl.h>

void loss_plot::add_loss_test(float loss_value)
{
	std::lock_guard<std::mutex> guard(m);
	loss_test.emplace_back(loss_value);
    if (loss_value > max_val)
        max_val = loss_value;
    new_data = true;
}

void loss_plot::add_loss_train(float loss_value)
{
	std::lock_guard<std::mutex> guard(m);
	loss_train.emplace_back(loss_value);
    if (loss_value > max_val)
        max_val = loss_value;
    new_data = true;
}

void loss_plot::draw()
{
    if (!new_data)
        return;
    new_data = false;
    mglGraph plot(0, options.w, options.h);

    mglData y_test(loss_test);
    y_test.Smooth("x");
    mglData y_train(loss_train);
    y_train.Smooth("x");
    plot.SetRanges(0, loss_train.size(), 0, max_val);
    //plot.SetOrigin(-1, -1, -1);  // first axis
    plot.Axis(); 
    plot.Label('y', "axis 1", 0);
    plot.Plot(y_train, "b");
    plot.AddLegend("train", "b");
    if(!loss_test.empty())
        plot.Plot(y_test, "r");
    plot.AddLegend("test", "r");
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
    loss_test.reserve(1000);
    loss_train.reserve(1000);
}