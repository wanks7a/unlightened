#include <NeuralNet.h>
#include <memory>

class gan
{
	std::vector<float> real_data_values;
	std::vector<float> fake_data_values;
	std::shared_ptr<NeuralNet> dis;
	std::shared_ptr<NeuralNet> gen;
	double loss_dis;
	double loss_gen;
	void generate_real_data();
	void generate_fake_data();
public:
	std::shared_ptr<NeuralNet>& discriminator();
	std::shared_ptr<NeuralNet>& generator();
	void set_discriminator(NeuralNet* disc);
	void set_discriminator(std::shared_ptr<NeuralNet>& disc);
	void set_generator(NeuralNet* gen);
	void set_generator(std::shared_ptr<NeuralNet>& gen);
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