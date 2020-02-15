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
	for (auto& f: filters)
	{
		conv2d_kernel(input_layer->getOutput(), input_layer->get_shape(), f.get_weights().get(), output.get(), output_shape, static_cast<unsigned int>(options.w));
	}
}

void cnn_layer::backprop(Layer* layer)
{
	shape filter_shape;
	filter_shape.width = options.w;
	filter_shape.height = options.h;
	filter_shape.depth = options.channels;
	for (auto& f : filters)
	{
		conv2d_kernel(input_layer->getOutput(), input_layer->get_shape(), layer->derivativeWithRespectToInput(), nullptr, filter_shape, static_cast<unsigned int>(options.w));
	}
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