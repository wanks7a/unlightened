#include <max_pool.h>

max_pool::max_pool(int f_size)
{
	device_layer = true;
	filter_size = f_size;
}

void max_pool::init(const shape& input)
{
	int add_one = 0;
	if ((input_shape.width % filter_size) > 0)
	{
		add_one = 1;
	}

	output_shape.width = input_shape.width / filter_size + add_one;

	add_one = 0;
	if ((input_shape.height % filter_size) > 0)
	{
		add_one = 1;
	}

	output_shape.height = input_shape.height / filter_size + add_one;

	output_shape.depth = input_shape.depth;
	output_shape.batches = input_shape.batches;

	mask.resize(output_shape.size());
	output.resize(output_shape.size());
	derivative.resize(input_shape.size());
}

void  max_pool::forward_pass(Layer* prevLayer)
{
	if (prevLayer->is_device_layer())
		max_pooling(prevLayer->get_output(), input_shape, output.get(), output_shape, mask.get(), filter_size);
	else
	{
		prevLayer->get_device_output(input);
		max_pooling(input.get(), input_shape, output.get(), output_shape, mask.get(), filter_size);
	}
}

void  max_pool::backprop(Layer* layer)
{
	if (layer->is_device_layer())
		max_pooling_backprop(layer->derivative_wr_to_input(), output_shape, derivative.get(), input_shape, mask.get(), filter_size);
	else
	{
		layer->get_device_output(derivative_input);
		max_pooling_backprop(derivative_input.get(), output_shape, derivative.get(), input_shape, mask.get(), filter_size);
	}
}

const float* max_pool::get_output()
{
	return output.get();
}

const float* max_pool::derivative_wr_to_input()
{
	return derivative.get();
}