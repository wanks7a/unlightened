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
public:
	virtual void init(Layer* layer) = 0;
	virtual void update_weights(Layer* layer) = 0;
	virtual ~optimizer() = default;
};

template <typename Device, typename DType>
class optimizer_device : public optimizer
{
	using optimizer::update_weights;
public:
	void update_weights(Layer* layer) override
	{
		auto weights_props = layer->get_weights();
		auto weights_deriv = layer->get_weights_deriv();
		if (!layer->is_device_layer())
		{
			device_vector<Device, DType> buffer1;
			device_vector<Device, DType> buffer2;
			buffer1.set_data(weights_props.ptr, weights_props.size);
			buffer2.set_data(weights_deriv.ptr, weights_deriv.size);
			update_weights(buffer1.data(), buffer2.data(), layer->get_learning_rate());
			auto v = buffer1.to_vector();
			memcpy((void*)weights_props.ptr, (const void*)v.data(), weights_props.size * sizeof(DType));
		}
		else
		{
			update_weights(weights_props.ptr, weights_deriv.ptr, layer->get_learning_rate());
		}
	}
	virtual ~optimizer_device() = default;
};