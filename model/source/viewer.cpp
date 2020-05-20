#include <viewer.h>
#include <iostream>

view::view(int w, int h, const std::string& name)
{
    options.w = w;
    options.h = h;
    options.name = name;
    if (!SDL_WasInit(SDL_INIT_VIDEO))
    {
        SDL_Init(SDL_INIT_VIDEO);
    }
}

void view::borderless(bool flag)
{
    options.is_borderless = flag;
}

void view::view_only(bool flag)
{
    options.view_only = flag;
}

bool view::is_view_only() const 
{
    return options.view_only;
}

SDL_Window* view::get_window() const
{
    return window;
}

int view::get_window_id() const
{
    return window_id;
}

void view::set_w_and_h(int w, int h)
{
    options.w = w;
    options.h = h;
}

bool view::show(int start_x, int start_y)
{
    int flags = SDL_WINDOW_OPENGL;
    if (options.is_borderless)
        flags |= SDL_WINDOW_BORDERLESS;
    window = SDL_CreateWindow(
        options.name.c_str(),                  // window title
        start_x,           // initial x position
        start_y,           // initial y position
        options.w,                               // width, in pixels
        options.h,                               // height, in pixels
        flags                  // flags - see below
    );

    if (!window)
        return false;

    window_id = SDL_GetWindowID(window);

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);

    return true;
}
void view::draw()
{
    if (renderer != nullptr)
        SDL_RenderPresent(renderer);
    else
        std::cout << "render == null" << std::endl;
}

view::~view()
{
    if (renderer != nullptr)
        SDL_DestroyRenderer(renderer);

    if (window != nullptr)
        SDL_DestroyWindow(window);
}

void viewer::loop()
{
    for (size_t i = 0; i < views.size(); i++)
    {
        views[i]->show(0, 0);
    }
    SDL_Event e;
    bool quit = false;
    while (!quit)
    {
        while (SDL_PollEvent(&e) != 0)
        {
            if (e.type == SDL_QUIT)
            {
                return;
            }
            execute_state(e);
        }
        for (size_t i = 0; i < views.size(); i++)
        {
            views[i]->draw();
        }
        SDL_Delay(30);
    }
}

void viewer::execute_state(const SDL_Event& e)
{
    switch (current_state)
    {
        case state::idle: idle_state(e); break;
        case state::mouse_left_down: break;
        case state::mouse_left_up: break;
        case state::start_moving: start_moving_window_state(e); break;
        case state::move_window: move_window_state(e); break;
    }
}

void viewer::move_window_state(const SDL_Event& e)
{
    if (e.type == SDL_MOUSEBUTTONUP || e.type == SDL_MOUSEMOTION)
    {
        view& v = get_view(e.button.windowID);
        int x, y;
        SDL_GetWindowPosition(v.get_window(), &x, &y);
        int add_x, add_y;
        add_x = e.button.x - last_mouse_position.x;
        add_y = e.button.y - last_mouse_position.y;
        SDL_SetWindowPosition(v.get_window(), x + add_x, y + add_y);
    }
    if (e.type == SDL_MOUSEBUTTONUP)
        current_state = state::idle;
}

void viewer::idle_state(const SDL_Event& e)
{
    if (e.type == SDL_MOUSEBUTTONDOWN 
        && e.button.state == SDL_PRESSED
        && e.button.button == SDL_BUTTON_LEFT)
    {
        auto& v = get_view(e.button.windowID);
        if (v.is_view_only())
        {
            current_state = state::start_moving;
            start_moving_window_state(e);
        }
        else
        {
            current_state = state::mouse_left_down;
        }
    }
}

view& viewer::get_view(int window_id)
{
    for (size_t i = 0; i < views.size(); i++)
    {
        if (views[i]->get_window_id() == window_id)
            return *views[i];
    }
    return *views[0];
}

void viewer::start_moving_window_state(const SDL_Event& e)
{
    last_mouse_position = e.button;
    current_state = state::move_window;
    move_window_state(e);
}

void viewer::add_view(std::shared_ptr<view>& v)
{
    views.emplace_back(v);
}

void viewer::add_view(view* v)
{
    views.emplace_back(std::shared_ptr<view>(v));
}