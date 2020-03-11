#include <cnn_layer.h>

cnn_layer::cnn_layer(size_t filter_dimension, size_t num_of_filters) : options(filter_dimension, filter_dimension, num_of_filters), filters_size(num_of_filters), input_layer(nullptr)
{
	device_layer = true;
}

void cnn_layer::init(const shape& input)
{
	options.channels = input.depth;
	filters.set_options(options);
	filters.init(input);	
	output_shape = filters.get_output_shape();
	output.resize(output_shape.size());
	input_derivative.resize(input.size());
	bias.resize(filters_size, 1.0f);
}

void cnn_layer::forward_pass(Layer* prevLayer)
{
	input_layer = prevLayer;

	if (prevLayer->is_device_layer())
		input = prevLayer->get_output();
	else
	{
		layer_input = prevLayer->get_device_output();
		input = layer_input.get();
	}

	for (int i = 0; i < filters.size(); i++)
	{
		conv_3d(input, input_layer->get_shape(), output.get() + i * output_shape.area(), output_shape, filters[i], bias.get(), options.w, options.h, options.w - filters.get_padding());
	}
}

void cnn_layer::backprop(Layer* layer)
{
	input_derivative.memset(0);

	size_t filter_step = output_shape.width * output_shape.height;
	int i = 0;
	cuVector<float> filter_output;
	cuVector<float> layer_input;
	const float* derivative = nullptr;

	if (layer->is_device_layer())
		derivative = layer->derivative_wr_to_input();
	else
	{
		layer_input = layer->get_device_derivative();
		derivative = layer_input.get();
	}

	shape filter_derivative_shape = filters.get_weights_derivative_shape();
	shape input_shape_temp = input_shape;
	input_shape.batches = 1;
	// calculate the derivative for the weights
	// output depth should equals number of filters
	for (size_t i = 0; i < options.num_of_filters; i++)
	{
		backprop_weights_3d(input, input_layer->get_shape(), filters.get_derivative(i), filter_derivative_shape,
			derivative + i * output_shape.area(),
			output_shape.width,
			output_shape.height, options.w - filters.get_padding(), output_shape.volume());
	}
	// now calculate the derivative for the input input

	cuVector<float> weights_flipped = cuVector<float>::from_device_to_device(filters.get_weights());
	shape filter_shape = filters.get_filter_shape();
	filter_shape.depth = filter_shape.depth * options.num_of_filters;
	flip_filter(weights_flipped.get(), filter_shape, false);
	flip_filter(weights_flipped.get(), filter_shape, true);
	filter_shape = filters.get_filter_shape();
	filter_shape.batches = options.num_of_filters;
	std::vector<float> result = cuVector<float>::from_device_host(weights_flipped.get(), weights_flipped.size());
	for (size_t i = 0; i < input_shape.depth; i++)
	{
		derivative_input_3d(weights_flipped.get(), filter_shape, input_derivative.get(), input_shape,
			derivative, output_shape.width, output_shape.height, options.w - filters.get_padding(), output_shape.volume());
	}

}

const float* cnn_layer::get_output()
{
	return output.get();
}

const float* cnn_layer::derivative_wr_to_input()
{
	return input_derivative.get();
}

void cnn_layer::printLayer()
{
}