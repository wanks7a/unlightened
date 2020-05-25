#include <reshape_layer.h>

reshape_layer::reshape_layer(shape out_shape) : output(nullptr), derivative(nullptr)
{
	output_shape = out_shape;
}

void reshape_layer::init(const shape& input)
{
	if (input.volume() != output_shape.volume())
	{
		std::cout << "Reshape: input shape volume is not the same as output volume !" << std::endl;
	}
	if (input.batches != output_shape.batches)
	{
		std::cout << "Reshape: input batch size is not the same as output batch size!" << std::endl;
	}
}

void reshape_layer::forward_pass(Layer* prevLayer)
{
	device_layer = prevLayer->is_device_layer();
	output = prevLayer->get_output();
}

void reshape_layer::backprop(Layer* layer)
{
	device_layer = layer->is_device_layer();
	derivative = layer->derivative_wr_to_input();
}

const float* reshape_layer::get_output()
{
	return output;
}

const float* reshape_layer::derivative_wr_to_input()
{
	return derivative;
}