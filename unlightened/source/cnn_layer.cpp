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
	output_shape.batches = input.batches;
	output.resize(output_shape.size());
	input_derivative.resize(input.size());
	bias.resize(filters_size, 1.0f);
}

void cnn_layer::forward_pass(Layer* prevLayer)
{
	input_layer = prevLayer;
	size_t filter_step = output_shape.width * output_shape.height;
	int i = 0;
	cuVector<float> filter_output;
	cuVector<float> layer_input;
	const float* input = nullptr;
	if (prevLayer->is_device_layer())
		input = prevLayer->get_output();
	else
	{
		layer_input = prevLayer->get_device_output();
		input = layer_input.get();
	}

	for (int i = 0; i < filters.size(); i++)
	{
		auto& filter = filters[i];
		auto filter_shape = filter.get_output_shape();
		filter_output.resize(filter_shape.size());
		conv_3d(input, input_shape, filter_output.get(), filter_shape, filter.get_weights().get(), filter.get_options().w, filter.get_options().zeropadding);
		merge_conv_with_bias(filter_output.get(), filter_shape, bias.get(), output.get() + i*filter_shape.area(), output_shape.volume());
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

	shape filter_shape(filters.back().get_options().w, filters.back().get_options().h, filters.back().get_options().channels);
	for (size_t batch = 0; batch < input_shape.batches; batch++)
	{
		for (size_t i = 0; i < filters.size(); i++)
		{
			cuVector<float> weights_rotated = cuVector<float>::from_device_to_device(filters[i].get_weights());
			flip_filter(weights_rotated.get(), filter_shape, true);
			flip_filter(weights_rotated.get(), filter_shape, false);
			full_conv_2d(weights_rotated.get(), filter_shape, input_derivative.get() + batch * input_shape.volume(), input_shape, derivative + output_shape.volume(), output_shape.width, true);
		}
	}

	flip_filter(input_derivative.get(), input_shape, false);
	flip_filter(input_derivative.get(), input_shape, true);

	const float* input = nullptr;
	if (input_layer->is_device_layer())
		input = input_layer->get_output();
	else
	{
		layer_input = input_layer->get_device_output();
		input = layer_input.get();
	}

	for (size_t batch = 0; batch < input_shape.batches; batch++)
	{
		for (size_t i = 0; i < filters.size(); i++)
		{
			conv_3d(input, input_shape, filters[i].get_weights_derivative().get(), filter_shape, derivative + batch * output_shape.volume() + i * output_shape.area(), output_shape.width, false);
			update_weights(filters[i].get_weights().get(), filters[i].get_weights_derivative().get(), filter_shape, learing_rate);
		}
	}
}

const float* cnn_layer::get_output()
{
	return output.get();
}

const float* cnn_layer::derivative_wr_to_input()
{
	return nullptr;
}

void cnn_layer::printLayer()
{
}