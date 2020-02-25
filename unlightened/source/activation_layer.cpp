#include <activation_layer.h>

activation_layer::activation_layer(activation_function function) : activ_func(function)
{
	device_layer = true;
}

void activation_layer::init(const shape& input)
{
	output_shape = input;
	output.resize(output_shape.size());
	derivative.resize(output_shape.size());
}

const float* activation_layer::get_output()
{
	return output.get();
}

const float* activation_layer::derivative_wr_to_input()
{
	return derivative.get();
}

void activation_layer::printLayer()
{

}

