#include <gan.h>

void gan::set_discriminator(NeuralNet* discriminator)
{
	dis = std::shared_ptr<NeuralNet>(discriminator);
	generate_real_data();
	generate_fake_data();
}

void gan::set_discriminator(std::shared_ptr<NeuralNet>& disc)
{
	dis = disc;
	generate_real_data();
	generate_fake_data();
}

void gan::set_generator(NeuralNet* generator)
{
	gen = std::shared_ptr<NeuralNet>(generator);
};

void gan::set_generator(std::shared_ptr<NeuralNet>& generator)
{
	gen = generator;
};

std::shared_ptr<NeuralNet>& gan::discriminator()
{
	return dis;
}

std::shared_ptr<NeuralNet>& gan::generator()
{
	return gen;
}

void gan::predict(const std::vector<float>& discrminator_real_data, const std::vector<float>& generator_noise)
{
	if (dis->getInputLayer().get_shape().size() != discrminator_real_data.size() ||
		gen->getInputLayer().get_shape().size() != generator_noise.size())
	{
		std::cout << "gan::predict input size is not matching" << std::endl;
		return;
	}
	loss_dis = 0;
	loss_gen = 0;
	dis->getInputLayer().set_input(discrminator_real_data);
	gen->getInputLayer().set_input(generator_noise);
	dis->predict();
	gen->predict();
	dis->loss_layer().setObservedValue(real_data_values);
	loss_dis += dis->loss_layer().get_total_loss();
	dis->loss_layer().print_predicted(2);
	dis->backprop(); // backprop discriminator so we can forward pass again with fake data 
	dis->getInputLayer().set_input(gen->loss_layer().get_output(), gen->loss_layer().get_shape().size());
	dis->predict();
	dis->loss_layer().setObservedValue(fake_data_values);
	dis->loss_layer().print_predicted(2);
	loss_dis += dis->loss_layer().get_total_loss();
}

void gan::backprop()
{
	dis->backprop(); // backprop with fake results so descriminator can detect them
	dis->set_update_weights(false); // forbid weights update so we dont lie the discriminator. This is done just to calculate the loss for the gan.
	dis->loss_layer().setObservedValue(real_data_values);
	loss_gen += dis->loss_layer().get_total_loss();
	dis->backprop();
	gen->loss_layer().set_derivative_manual((*dis)[0]->get_native_derivative());
	gen->backprop(false);
	dis->set_update_weights(true);
}

void gan::generate_real_data()
{
	for (size_t i = 0; i < dis->getInputLayer().get_shape().batches; i++)
	{
		real_data_values.emplace_back(1.0f);
		real_data_values.emplace_back(0.0f);
	}
}

void gan::generate_fake_data()
{
	for (size_t i = 0; i < dis->getInputLayer().get_shape().batches; i++)
	{
		fake_data_values.emplace_back(0.0f);
		fake_data_values.emplace_back(1.0f);
	}
}