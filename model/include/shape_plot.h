#include <viewer.h>
#include <mutex>
#include <shape.h>

class shape_plot : public view
{
	shape sh;
	std::vector<float> data;
	std::mutex m;
	bool new_data = false;
public:
	shape_plot(int w, int h, std::string name);
	void draw() override;
	void set_shape_data(std::vector<float>&& data, const shape& sh);
};