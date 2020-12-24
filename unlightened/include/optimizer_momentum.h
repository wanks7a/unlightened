#pragma once
#include <optimizer.h>
#include <cuda_device.h>
#include <device_vector.h>
#include <cuda_device.h>
#include <cudnn_helpers.h>

class momentum_optimizer : public optimizer_device<cuda_device, float>
{
	float beta;
	device_vector<cuda_device, float> momentum_derivatives;
	cudnn_hanlde handle;
	cudnn_descriptor4d weights_desc;
	cudnn_add_tensor add_tensor;
	void update_weights(float* weights, const float* deriv, float learning_rate) override;
public:
	using optimizer_device::update_weights;
	momentum_optimizer(float beta = 0.9f) : beta(beta) {}
	void init(Layer* layer) override;
	virtual ~momentum_optimizer() = default;
};