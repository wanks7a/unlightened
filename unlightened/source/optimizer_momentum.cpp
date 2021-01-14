#include <optimizer_momentum.h>
#include <cuda_vector_norm.h>

void momentum_optimizer::update_weights(float* weights, const float* deriv, float learning_rate)
{
	device_vector<cuda_device, float> deriv_copy;
	if (clip_grads_norm > 0.0f)
	{
		deriv_copy.memcpy(deriv, weights_desc.sh.size());
		float norm = cuda_vector_norm(deriv_copy.data(), deriv_copy.size());
		if (norm > clip_grads_norm)
		{
			cuda_scale_vector(deriv_copy.data(), deriv_copy.size(), clip_grads_norm / norm);
			deriv = deriv_copy.data();
		}
	}

	float one = 1.0f;
	float zero = 0.0f;
	CUDA_CHECK(cudnnOpTensor(handle.handle,
		add_tensor.descriptor,
		&gamma,
		weights_desc.descriptor,
		momentum_derivatives.data(),
		&one,
		weights_desc.descriptor,
		deriv,
		&zero,
		weights_desc.descriptor,
		momentum_derivatives.data()));
	learning_rate = -(learning_rate / sh.batches);
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
	device_vector<cuda_device, float> deriv_copy;
	if (clip_grads_norm > 0.0f)
	{
		deriv_copy.memcpy(deriv, bias_desc.sh.size());
		float norm = cuda_vector_norm(deriv_copy.data(), deriv_copy.size());
		if (norm > clip_grads_norm)
		{
			cuda_scale_vector(deriv_copy.data(), deriv_copy.size(), clip_grads_norm / norm);
			deriv = deriv_copy.data();
		}
	}

	float one = 1.0f;
	float zero = 0;
	CUDA_CHECK(cudnnOpTensor(handle.handle,
		add_tensor.descriptor,
		&gamma,
		bias_desc.descriptor,
		momentum_derivatives_bias.data(),
		&one,
		bias_desc.descriptor,
		deriv,
		&zero,
		bias_desc.descriptor,
		momentum_derivatives_bias.data()));
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
	sh = layer->get_shape();
	if (weight_props.size == 0)
		return;
	momentum_derivatives.resize(weight_props.size, 0.0f);
	if (!weights_desc.create(weight_props.size, 1, 1, 1))
		std::exit(1);
	add_tensor.create();
	auto bias_props = layer->get_bias();
	if (bias_props.size == 0)
		return;
	momentum_derivatives_bias.resize(bias_props.size, 0.0f);
	bias_desc.create(bias_props.size, 1, 1, 1);
}

void momentum_optimizer::pre_epoch(size_t e)
{
	//if(momentum_derivatives.size() > 0)
	//	cuda_scale_vector(momentum_derivatives.data(), momentum_derivatives.size(), 0.0f);
	//if(momentum_derivatives_bias.size() > 0)
	//	cuda_scale_vector(momentum_derivatives_bias.data(), momentum_derivatives_bias.size(), 0.0f);
}

void momentum_optimizer::post_epoch(size_t e)
{

}