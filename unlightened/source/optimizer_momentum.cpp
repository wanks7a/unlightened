#include <optimizer_momentum.h>

void momentum_optimizer::update_weights(float* weights, const float* deriv, float learning_rate)
{
	float minus_beta = 1.0f - beta;
	float zero = 0;
	cudnnOpTensor(handle.handle,
		add_tensor.descriptor,
		&beta,
		weights_desc.descriptor,
		momentum_derivatives.data(),
		&minus_beta,
		weights_desc.descriptor,
		deriv,
		&zero,
		weights_desc.descriptor,
		momentum_derivatives.data());
	float one = 1.0f;
	learning_rate = -learning_rate;
	cudnnOpTensor(handle.handle,
		add_tensor.descriptor,
		&one,
		weights_desc.descriptor,
		weights,
		&learning_rate,
		weights_desc.descriptor,
		momentum_derivatives.data(),
		&zero,
		weights_desc.descriptor,
		weights);
}


void momentum_optimizer::init(Layer* layer)
{
	auto weight_props = layer->get_weights();
	momentum_derivatives.reserve(weight_props.size);
	weights_desc.create(weight_props.size, 1, 1, 1);
	add_tensor.create();
}