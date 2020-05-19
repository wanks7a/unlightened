#include <viewer.h>
#include <vector>
#include <mutex>

class loss_plot : public view
{
	std::mutex m;
	std::vector<float> loss_train;
	std::vector<float> loss_test;
	std::vector<int> train_examples;
	std::vector<int> loss_examples;
	float min_val = 0;
	float max_val = 0;
public:
	loss_plot();
	void add_loss_train(float value);
	void add_loss_test(float value);
	void draw() override;
};