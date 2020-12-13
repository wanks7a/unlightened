#include <NeuralNet.h>
#include <memory>

class gan
{
	std::vector<float> real_data_values;
	std::vector<float> fake_data_values;
	std::shared_ptr<model> dis;
	std::shared_ptr<model> gen;
	double loss_dis;
	double loss_gen;
	void generate_real_data();
	void generate_fake_data();
public:
	std::shared_ptr<model>& discriminator();
	std::shared_ptr<model>& generator();
	void set_discriminator(model* disc);
	void set_discriminator(std::shared_ptr<model>& disc);
	void set_generator(model* gen);
	void set_generator(std::shared_ptr<model>& gen);
	void predict(const std::vector<float>& discrminator_real_data, const std::vector<float>& generator_noise);
	void backprop();
	float discriminator_loss() const 
	{
		return loss_dis;
	}

	float generator_loss() const
	{
		return loss_gen;
	}
};