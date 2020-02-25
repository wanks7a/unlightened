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
	for (size_t b = 0; b < input_shape.batches; b++)
	{
		for (size_t i = 0; i < output_shape.depth; i++)
		{
			int filter_offset = i * filter_derivative_shape.volume() + b * filter_derivative_shape.volume() * output_shape.depth;
			int derivative_offset = i * output_shape.area() + b * output_shape.volume();

	/*		conv_3d(input + b * input_shape.volume(), input_shape_temp, filters.get_weights_derivative().get() + filter_offset, filter_derivative_shape, 
					derivative + derivative_offset, 
					nullptr, 
					output_shape.width, 
					output_shape.height, 
					options.w - filters.get_padding());
			std::vector<float> res;
			filters.get_weights_derivative().getCopy(res);*/
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