#include <cnn_layer.h>

cnn_layer::cnn_layer(size_t filter_dimension, size_t num_of_filters) : options(filter_dimension, filter_dimension), filters_size(num_of_filters), input_layer(nullptr)
{
}

void cnn_layer::init(const shape& input)
{
	options.channels = input.depth;
	for (size_t i = 0; i < filters_size; i++)
	{
		filters.emplace_back(options);
		filters.back().init(input);
	}
	output_shape = filters.back().get_output_shape();
	output_shape.depth = filters_size;
	output.resize(output_shape.size());
}

void cnn_layer::forwardPass(Layer* prevLayer)
{
	input_layer = prevLayer;
	size_t filter_step = output_shape.width * output_shape.height;
	int i = 0;
	for (auto& f: filters)
	{
		//conv2d_kernel(input_layer->getOutput(), input_layer->get_shape(), f.get_weights().get(), output.get() + filter_step*i, output_shape, static_cast<unsigned int>(options.w));
		i++;
	}
}

void cnn_layer::backprop(Layer* layer)
{
	shape filter_shape;
	filter_shape.width = options.w;
	filter_shape.height = options.h;
	filter_shape.depth = options.channels;
	int i = 0;
	size_t filter_step = output_shape.width * output_shape.height;
	// here is the backprop for the weights here we obtain the filter derivatives
	for (auto& f : filters)
	{
		//conv2d_kernel(input_layer->getOutput(), input_layer->get_shape(), layer->derivativeWithRespectToInput() + filter_step * i, f.get_weights_derivative().get(), filter_shape, static_cast<unsigned int>(options.w));
		i++;
	}
	// now obtain the derivatives with respect to the input
	// we need to rotate the filter to 180 degrees
	// after that apply full convolution over derivatives 
	// the output is the derivative with respect to the input layer
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