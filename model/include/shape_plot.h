#include <viewer.h>
#include <mutex>
#include <shape.h>
#include <functional>

class shape_plot : public view
{
	std::mutex m;
	bool new_data = false;
	float scale = 1.0f;
	std::function<void(shape_plot&)> latest_draw_call;
	
	struct data_properties
	{
		std::vector<float> data;
		size_t width = 0;
		size_t height = 0;
		float min_val = 0.0f;
		float max_val = 0.0f;
	};
	data_properties rgb_data;
	data_properties swap_data;
	void draw_rgb_channels_internal();
	void draw_grayscale_internal();
public:
	shape_plot(int w, int h, std::string name);
	void draw() override;
	bool draw_rgb_channels(std::vector<float>&& rgb_image, size_t width, size_t height, float min_val = 0.0f, float max_val = 255.0f);
	bool draw_grayscale(std::vector<float>&& image, size_t width, size_t height, float min_val = 0.0f, float max_val = 255.0f);
	void set_scale(float val);
};