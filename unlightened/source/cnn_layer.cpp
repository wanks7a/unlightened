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
		conv_3d(input, filter_shape, filter_output.get(), filter_shape, filter.get_weights().get(), filter.get_options().w, filter.get_options().zeropadding);
		merge_conv_with_bias(filter_output.get(), filter_shape, bias.get(), output.get() + i*filter_shape.area(), output_shape.volume());
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
		//conv2d_kernel(input_layer->get_output(), input_layer->get_shape(), layer->derivative_wr_to_input() + filter_step * i, f.get_weights_derivative().get(), filter_shape, static_cast<unsigned int>(options.w));
		i++;
	}
	// now obtain the derivatives with respect to the input
	// we need to rotate the filter to 180 degrees
	// after that apply full convolution over derivatives 
	// the output is the derivative with respect to the input layer
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