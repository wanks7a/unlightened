#pragma once
#include <Layer.h>
#include <cstring>
#include <device_vector.h>

enum OPTIMIZER
{
	Momentum
};

class optimizer
{
protected:
    virtual void update_weights(float* weights, const float* deriv, float learning_rate) = 0;
    virtual void update_bias(float* bias, const float* deriv, float learning_rate) = 0;
public:
	virtual void init(Layer* layer) = 0;
	virtual void update(Layer* layer) = 0;
	virtual ~optimizer() = default;
};

template <typename Device, typename DType>
class optimizer_device : public optimizer
{
	using optimizer::update_weights;
	void update_weights_internal(weights_properties& weights_props, weights_properties& weights_deriv, float learning_rate, bool is_device_memory)
	{
		if (weights_props.size == 0 || weights_deriv.size == 0)
			return;
		if (weights_props.size != weights_deriv.size)
			return;
		if (!is_device_memory)
		{
			device_vector<Device, DType> buffer1;
			device_vector<Device, DType> buffer2;
			buffer1.set_data(weights_props.ptr, weights_props.size);
			buffer2.set_data(weights_deriv.ptr, weights_deriv.size);
			update_weights(buffer1.data(), buffer2.data(), learning_rate);
			auto v = buffer1.to_vector();
			memcpy((void*)weights_props.ptr, (const void*)v.data(), weights_props.size * sizeof(DType));
		}
		else
		{
			update_weights(weights_props.ptr, weights_deriv.ptr, learning_rate);
		}
	}
	void update_bias_internal(weights_properties& weights_props, weights_properties& weights_deriv, float learning_rate, bool is_device_memory)
	{
		if (weights_props.size == 0 || weights_deriv.size == 0)
			return;
		if (weights_props.size != weights_deriv.size)
			return;
		if (!is_device_memory)
		{
			device_vector<Device, DType> buffer1;
			device_vector<Device, DType> buffer2;
			buffer1.set_data(weights_props.ptr, weights_props.size);
			buffer2.set_data(weights_deriv.ptr, weights_deriv.size);
			update_bias(buffer1.data(), buffer2.data(), learning_rate);
			auto v = buffer1.to_vector();
			memcpy((void*)weights_props.ptr, (const void*)v.data(), weights_props.size * sizeof(DType));
		}
		else
		{
			update_bias(weights_props.ptr, weights_deriv.ptr, learning_rate);
		}
	}
public:
	void update(Layer* layer) override
	{
		if (!layer->is_frozen())
			return;
		auto weights_props = layer->get_weights();
		auto weights_deriv = layer->get_weights_deriv();
		update_weights_internal(weights_props, weights_deriv, layer->get_learning_rate(), layer->is_device_layer());
		weights_props = layer->get_bias();
		weights_deriv = layer->get_bias_deriv();
		update_bias_internal(weights_props, weights_deriv, layer->get_learning_rate(), layer->is_device_layer());
	}
	virtual ~optimizer_device() = default;
};