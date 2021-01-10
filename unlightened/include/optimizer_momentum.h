#pragma once
#include <optimizer.h>
#include <cuda_device.h>
#include <device_vector.h>
#include <cuda_device.h>
#include <cudnn_helpers.h>

class momentum_optimizer : public optimizer_device<cuda_device, float>
{
	float beta;
	float clip_grads_norm;
	device_vector<cuda_device, float> momentum_derivatives;
	device_vector<cuda_device, float> momentum_derivatives_bias;
	cudnn_hanlde handle;
	cudnn_descriptor4d weights_desc;
	cudnn_descriptor4d bias_desc;
	cudnn_add_tensor add_tensor;
	void update_weights(float* weights, const float* deriv, float learning_rate) override;
	void update_bias(float* bias, const float* deriv, float learning_rate) override;
	shape sh;
public:
	using optimizer_device::update_weights;
	momentum_optimizer(float beta_param = 0.9f, float clip_grads_norm_param = 0.0f) : beta(beta_param), clip_grads_norm(clip_grads_norm_param) {}
	void init(Layer* layer) override;
	virtual ~momentum_optimizer() = default;
};