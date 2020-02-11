#include <cnn_layer.h>

cnn_layer::cnn_layer(size_t filter_dimension, size_t num_of_filters) : options(filter_dimension, filter_dimension)
{
}

void cnn_layer::init(const shape& input)
{
	
}

void cnn_layer::forwardPass(Layer* prevLayer)
{

}

void cnn_layer::backprop(Layer* layer)
{
}

const float* cnn_layer::getOutput()
{
	return nullptr;
}

const float* cnn_layer::derivativeWithRespectToInput()
{
	return nullptr;
}

void cnn_layer::printLayer()
{
}