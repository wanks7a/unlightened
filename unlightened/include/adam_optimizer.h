#pragma once
#include <optimizer.h>
#include <cuda_vector.h>

class adam_optimizer : public optimizer_device<cuda_device, float>
{
	float beta1, beta2;
	float beta1_tpower, beta2_tpower;
	float eps;
	cuda_floats m_weights;
	cuda_floats v_weights;
	cuda_floats m_bias;
	cuda_floats v_bias;
	size_t batches= 0;
	void update_weights(float* weights, const float* deriv, float learning_rate) override;
	void update_bias(float* bias, const float* deriv, float learning_rate) override;
	void pre_epoch(size_t e) override;
	void post_epoch(size_t e) override;
	void update_params(cuda_floats& m, cuda_floats& v, cuda_floats& w, cuda_floats& d, cuda_floats& m_hat, cuda_floats& v_hat, cuda_floats& workspace, float learning_rate);
public:
	adam_optimizer(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f) :  beta1(beta1), beta2(beta2), eps(epsilon), beta1_tpower(beta1), beta2_tpower(beta2){}
	void update(Layer* layer) override;
	void init(Layer* layer) override;
	virtual ~adam_optimizer() = default;
};