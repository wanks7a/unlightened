#include <activation_layer.h>

activation_layer::activation_layer(activation_function function) : activ_func(function)
{
}

void activation_layer::init(const shape& input)
{
	output_shape = input;
	output.resize(output_shape.size());
}

const float* activation_layer::getOutput()
{
	return output.get();
}

const float* activation_layer::derivativeWithRespectToInput()
{
	return nullptr;
}

void activation_layer::printLayer()
{

}

