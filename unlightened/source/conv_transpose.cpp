#include <conv_transpose.h>

conv2d_transposed::conv2d_transposed(const filter_options& opt) : conv2d_cudnn(0,0, false)
{
	options = opt;
}

void conv2d_transposed::init(const shape& input)
{
	shape out_sh;
	out_sh.width = (input.width - 1) + options.w; // this formula doesn't count for the stride
	out_sh.height = (input.height - 1) + options.h;
	out_sh.depth = options.channels;
	out_sh.batches = input.batches;
	options.num_of_filters = input.depth;
	input_shape = out_sh;
	filters.set_options(options);
	filters.init(out_sh);
	output_shape = filters.get_output_shape();
	output.resize(output_shape.size());
	input_derivative.resize(out_sh.size());
	filters.get_weights_derivative().resize(filters.get_filter_shape().size() * filters.get_options().num_of_filters);
	init_cudnn();
}

void conv2d_transposed::forward_pass(Layer* prevLayer)
{
	float alpha = 1.0f, alpha2 = 1.0f, beta = 0.0f;
	input_layer = prevLayer;

	if (prevLayer->is_device_layer())
		input = prevLayer->get_output();
	else
	{
		prevLayer->get_device_output(layer_input);
		input = layer_input.get();
	}
	
	backprop_cudnn(input);

	// add bias to the output from the convolution
	checkCUDNN(cudnnOpTensor(cudnn_handle,
		add_op_descriptor,
		&alpha,
		input_descriptor,
		input_derivative.get(),
		&alpha2,
		bias_tensor_descriptor,
		filters.get_bias().get(),
		&beta,
		input_descriptor,
		input_derivative.get()));
}

void conv2d_transposed::backprop(Layer* layer)
{

}

const float* conv2d_transposed::get_output()
{
	return input_derivative.get();
}

const float* conv2d_transposed::derivative_wr_to_input()
{
	return output.get();
}