#pragma once
#include <SDL.h>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <memory>

struct view_options
{
	int w = 1024;
	int h = 768;
	std::string name = "View";
	bool is_borderless = true;
	bool view_only = true;
};

class view
{
protected:
	SDL_Window* window = nullptr;
	SDL_Renderer* renderer = nullptr;
	int window_id = 0;
	view_options options;
public:
	view(int w = 1024, int h = 768, const std::string& name = "View");
	void borderless(bool flag);
	void view_only(bool flag);
	bool show(int start_x, int start_y);
	bool is_view_only() const;
	void set_w_and_h(int w, int h);
	virtual void draw();
	SDL_Window* get_window() const;
	int get_window_id() const;
	~view();
};

class viewer
{
	enum class state
	{
		idle = 0,
		mouse_left_down = 1,
		mouse_left_up = 2,
		start_moving = 3,
		move_window = 4,
	};
	SDL_MouseButtonEvent last_mouse_position;
	state current_state = state::idle;
	std::vector<std::shared_ptr<view>> views;

public:
	void loop();
	void add_view(std::shared_ptr<view>& v);
private:
	view& get_view(int window_id);

	void execute_state(const SDL_Event& e);
	void move_window_state(const SDL_Event& e);
	void idle_state(const SDL_Event& e);
	void start_moving_window_state(const SDL_Event& e);
};