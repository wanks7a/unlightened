#include <adam_optimizer.h>
#include <cuda_vector_norm.h>
#include <generic_functions.h>

void adam_optimizer::update_weights(float* weights, const float* deriv, float learning_rate)
{
	if (m_weights.size() > 4096 * 20)
		adam_kernel_vectorized(m_weights.data(), v_weights.data(), weights, deriv, m_weights.size(), beta1, beta2, beta1_tpower, beta2_tpower, learning_rate / batches);
	else
		adam_kernel(m_weights.data(), v_weights.data(), weights, deriv, m_weights.size(), beta1, beta2, beta1_tpower, beta2_tpower, learning_rate / batches);
}

void adam_optimizer::update_bias(float* bias, const float* deriv, float learning_rate)
{
	if (m_bias.size() > 4096 * 20)
		adam_kernel_vectorized(m_bias.data(), v_bias.data(), bias, deriv, m_bias.size(), beta1, beta2, beta1_tpower, beta2_tpower, learning_rate / batches);
	else
		adam_kernel(m_bias.data(), v_bias.data(), bias, deriv, m_bias.size(), beta1, beta2, beta1_tpower, beta2_tpower, learning_rate / batches);
}


void adam_optimizer::init(Layer* layer)
{
	auto weight_props = layer->get_weights();
	m_weights.resize(weight_props.size, 0.0f);
	v_weights.resize(weight_props.size, 0.0f);
	weight_props = layer->get_bias();
	m_bias.resize(weight_props.size, 0.0f);
	v_bias.resize(weight_props.size, 0.0f);
}

void adam_optimizer::pre_epoch(size_t e)
{
}

void adam_optimizer::post_epoch(size_t e)
{
}

void adam_optimizer::update(Layer* layer)
{
	batches = layer->get_shape().batches;
	optimizer_device::update(layer);
	beta1_tpower *= beta1;
	beta2_tpower *= beta2;
}