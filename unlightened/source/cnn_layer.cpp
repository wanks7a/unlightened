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
		conv_3d(input, input_layer->get_shape(), output.get() + i * output_shape.area(), output_shape, filters[i], filters.get_bias().get(), options.w, options.h, options.w - filters.get_padding());
	}
}

void cnn_layer::backprop(Layer* layer)
{
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
	// calculate the derivative for the weights
	// output depth should equals number of filters
	for (size_t i = 0; i < options.num_of_filters; i++)
	{
		backprop_weights_3d(input, input_layer->get_shape(), filters.get_derivative(i), filter_derivative_shape,
			derivative + i * output_shape.area(),
			output_shape.width,
			output_shape.height, output_shape.width - filters.get_padding(), output_shape.volume());
	}
	// now calculate the derivative for the input input

	cuVector<float> weights_flipped = cuVector<float>::from_device_to_device(filters.get_weights());
	shape filter_shape = filters.get_filter_shape();
	filter_shape.depth = filter_shape.depth * options.num_of_filters;
	flip_filter(weights_flipped.get(), filter_shape, false);
	flip_filter(weights_flipped.get(), filter_shape, true);
	filter_shape = filters.get_filter_shape();
	filter_shape.batches = options.num_of_filters;

	derivative_input_3d(weights_flipped.get(), filter_shape, input_derivative.get(), input_shape,
		derivative, output_shape.width, output_shape.height, output_shape.width - filters.get_padding(), output_shape.volume());

	update_bias(derivative, output_shape, filters.get_bias().get(), learing_rate);
	//auto old_weights = filters.get_weights().to_vector();
	update_weights(filters.get_weights_derivative().get(), filters.get_weights_derivative_shape(), filters.size(), filters.get_weights().get(), learing_rate);
	//auto new_weights = filters.get_weights().to_vector();
	//auto inp_deriv = layer->get_native_derivative();
	//auto filter_weights_deriv = filters.get_weights_derivative().to_vector();
	//auto input_deriv2 = input_derivative.to_vector();
	//std::vector<float> differnce;
	//for (size_t i = 0; i < old_weights.size(); i++)
	//{
	//	if((old_weights[i] - new_weights[i]) != 0)
	//		differnce.emplace_back(old_weights[i] - new_weights[i]);
	//}
}

const float* cnn_layer::get_output()
{
	return output.get();
}

const float* cnn_layer::derivative_wr_to_input()
{
	return input_derivative.get();
}