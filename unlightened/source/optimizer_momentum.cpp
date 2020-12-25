#include <optimizer_momentum.h>

void momentum_optimizer::update_weights(float* weights, const float* deriv, float learning_rate)
{
	float minus_beta = 1.0f - beta;
	float zero = 0;
	device_vector<cuda_device, float> w;
	w.set_data(weights, weights_desc.sh.size());
	auto w1 = w.to_vector();
	CUDA_CHECK(cudnnOpTensor(handle.handle,
		add_tensor.descriptor,
		&beta,
		weights_desc.descriptor,
		momentum_derivatives.data(),
		&minus_beta,
		weights_desc.descriptor,
		deriv,
		&zero,
		weights_desc.descriptor,
		momentum_derivatives.data()));
	device_vector<cuda_device, float> m;
	m.set_data(momentum_derivatives.data(), weights_desc.sh.size());
	auto m1 = m.to_vector();
	float one = 1.0f;
	learning_rate = -learning_rate / sh.batches;
	CUDA_CHECK(cudnnOpTensor(handle.handle,
		add_tensor.descriptor,
		&one,
		weights_desc.descriptor,
		weights,
		&learning_rate,
		weights_desc.descriptor,
		momentum_derivatives.data(),
		&zero,
		weights_desc.descriptor,
		weights));
}

void momentum_optimizer::update_bias(float* bias, const float* deriv, float learning_rate)
{
	float minus_beta = 1.0f - beta;
	float zero = 0;
	CUDA_CHECK(cudnnOpTensor(handle.handle,
		add_tensor.descriptor,
		&beta,
		bias_desc.descriptor,
		momentum_derivatives_bias.data(),
		&minus_beta,
		bias_desc.descriptor,
		deriv,
		&zero,
		bias_desc.descriptor,
		momentum_derivatives_bias.data()));
	float one = 1.0f;
	learning_rate = -learning_rate / sh.batches;
	CUDA_CHECK(cudnnOpTensor(handle.handle,
		add_tensor.descriptor,
		&one,
		bias_desc.descriptor,
		bias,
		&learning_rate,
		bias_desc.descriptor,
		momentum_derivatives_bias.data(),
		&zero,
		bias_desc.descriptor,
		bias));
}


void momentum_optimizer::init(Layer* layer)
{
	auto weight_props = layer->get_weights();
	momentum_derivatives.reserve(weight_props.size);
	weights_desc.create(weight_props.size, 1, 1, 1);
	add_tensor.create();
	auto bias_props = layer->get_bias();
	momentum_derivatives_bias.reserve(bias_props.size);
	bias_desc.create(bias_props.size, 1, 1, 1);
	sh = layer->get_shape();
}