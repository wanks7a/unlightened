#include <viewer.h>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <mgl2/mgl.h>
#include <memory>

class loss_plot : public view
{
	struct plot_values
	{
		std::string color;
		std::vector<float> points;
	};
	bool new_data = false;
	std::mutex m;
	float max_val = 0.0f;
	SDL_Texture* last_texture = nullptr;
	std::unordered_map<std::string, plot_values> lines;
public:
	loss_plot(int w, int h, std::string name);
	void add_point(float point, const std::string& name, const std::string color);
	void add_points(const std::vector<float>& points, const std::string& name, const std::string color);
	void draw() override;
};