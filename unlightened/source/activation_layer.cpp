#include <activation_layer.h>

activation_layer::activation_layer() : activ_func(activation_function::Sigmoid)
{
	device_layer = true;
}


activation_layer::activation_layer(activation_function function) : activ_func(function)
{
	device_layer = true;
}

void activation_layer::init(const shape& input)
{
	output_shape = input;
	output.resize(output_shape.size());
	derivative.resize(output_shape.size());
	if (activ_func == activation_function::Softmax)
	{
		softmax.init(input);
	}
}

const float* activation_layer::get_output() const
{
	return output.get();
}

const float* activation_layer::derivative_wr_to_input() const
{
	return derivative.get(); 
}