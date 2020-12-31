#include <conv_transpose.h>
#include <algorithm>

conv2d_transposed::conv2d_transposed() : pad_type(padding::SAME)
{
	in_device_memory = true;
}

conv2d_transposed::conv2d_transposed(size_t number_filters, size_t filter_size, size_t stride, padding pad) : pad_type(pad)
{
	in_device_memory = true;
	filter_options options;
	options.w = filter_size;
	options.h = filter_size;
	options.channels = number_filters;
	options.stride = stride;
	if (pad_type == padding::SAME)
		options.zeropadding = true;
	else
		options.zeropadding = false;
	filter_data.set_options(options);
}

void conv2d_transposed::init(const shape& input)
{
	filter_options options = filter_data.get_options();
	options.num_of_filters = input.depth;
	filter_data.set_options(options);

	output_shape.width = calc_output_dim(input.width);
	output_shape.height = calc_output_dim(input.height);
	output_shape.depth = options.channels;
	output_shape.batches = input.batches;
	output.resize(output_shape.size());
	filter_data.init(output_shape);
	deriv_output.resize(input_shape.size());
	filter_data.get_weights_derivative().resize(filter_data.get_filter_shape().size() * filter_data.get_options().num_of_filters);
	init_tensors();
}

void conv2d_transposed::init_tensors()
{
	bool initialized = true;
	initialized &= input_desc.create(input_shape.width, input_shape.height, input_shape.depth, input_shape.batches);
	initialized &= out_desc.create(output_shape.width, output_shape.height, output_shape.depth, output_shape.batches);
	initialized &= bias_desc.create(1, 1, filter_data.get_options().num_of_filters, 1);
	initialized &= weight_desc.create(filter_data.get_options().w, filter_data.get_options().h, filter_data.get_options().channels, filter_data.get_options().num_of_filters);
	initialized &= filter_desc.create(filter_data.get_options().w, filter_data.get_options().h, filter_data.get_options().channels, filter_data.get_options().num_of_filters);
	initialized &= conv2d_desc.create(filter_data.get_padding(), filter_data.get_padding(), filter_data.get_options().stride);
	initialized &= add_tensor.create();

	initialized &= cudnn_status_check(cudnnGetConvolutionForwardAlgorithm(handle.handle,
		out_desc.descriptor,
		filter_desc.descriptor,
		conv2d_desc.descriptor,
		input_desc.descriptor,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		/*memoryLimitInBytes=*/0,
		&conv2d_forwardpass_alg));
	size_t max_mem = 0;
	size_t workspace_bytes = 0;
	initialized &= cudnn_status_check(cudnnGetConvolutionForwardWorkspaceSize(handle.handle,
		out_desc.descriptor,
		filter_desc.descriptor,
		conv2d_desc.descriptor,
		input_desc.descriptor,
		conv2d_forwardpass_alg,
		&workspace_bytes));

	max_mem = std::max(workspace_bytes, max_mem);
	workspace_bytes = 0;

	initialized &= cudnn_status_check(cudnnGetConvolutionBackwardDataWorkspaceSize(handle.handle,
		filter_desc.descriptor,
		input_desc.descriptor,
		conv2d_desc.descriptor,
		out_desc.descriptor,
		backprop_algo,
		&workspace_bytes));

	max_mem = std::max(workspace_bytes, max_mem);
	workspace_bytes = 0;

	cudnnConvolutionBwdFilterPreference_t algo_pref = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;

	initialized &= cudnn_status_check(cudnnGetConvolutionBackwardFilterAlgorithm(handle.handle,
		out_desc.descriptor,
		input_desc.descriptor,
		conv2d_desc.descriptor,
		filter_desc.descriptor,
		algo_pref,
		0,
		&filter_backprop_algo));

	initialized &= cudnn_status_check(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle.handle,
		out_desc.descriptor,
		input_desc.descriptor,
		conv2d_desc.descriptor,
		filter_desc.descriptor,
		filter_backprop_algo,
		&workspace_bytes
	));
	max_mem = std::max(workspace_bytes, max_mem);

	shared_mem.resize(max_mem / sizeof(float));

	if (!initialized || !handle)
	{
		// to do error
		std::exit(EXIT_FAILURE);
	}
}


void conv2d_transposed::forward_pass(Layer* layer)
{
	const float* derivative = nullptr;

	if (layer->is_device_layer())
		derivative = layer->get_output();
	else
	{
		layer_input = layer->get_device_output();
		derivative = layer_input.get();
	}
	input_from_prev_layer = derivative;
	backprop_cudnn(derivative);
	// TO DO BIAS
}

void conv2d_transposed::backprop(Layer* layer)
{
	cuVector<float> deriv;
	const float* deriv_ptr = nullptr;

	if (layer->is_device_layer())
		deriv_ptr = layer->derivative_wr_to_input();
	else
	{
		deriv = layer->get_device_derivative();
		deriv_ptr = deriv.get();
	}

	const float alpha = 1.0f, alpha2 = 1.0f, beta = 0.0f;

	// convolution 
	backprop_weights_cudnn(deriv_ptr, input_from_prev_layer);

	//if (!is_first_layer)
	//{
		cudnnConvolutionForward(handle.handle,
			&alpha,
			out_desc.descriptor,
			deriv_ptr,
			filter_desc.descriptor,
			filter_data.get_weights().get(),
			conv2d_desc.descriptor,
			conv2d_forwardpass_alg,
			shared_mem.get(),
			shared_mem.size() * sizeof(float),
			&beta,
			input_desc.descriptor,
			deriv_output.get());
	//}
	if (update_on_backprop)
	{
		//update_weights();
		//update_bias(derivative, output_shape, filters.get_bias().get(), learing_rate);
	}
}

const float* conv2d_transposed::get_output() const
{
	return output.get();
}

const float* conv2d_transposed::derivative_wr_to_input() const
{
	return deriv_output.get();
}

size_t conv2d_transposed::calc_output_dim(size_t input) const
{
	input *= filter_data.get_options().stride;
	if (pad_type == padding::VALID)
	{
		input += std::max(static_cast<int>(filter_data.get_options().w - filter_data.get_options().stride), 0);
	}
	else if(pad_type == padding::FULL)
	{
		input -= (filter_data.get_options().w - filter_data.get_options().stride - 2);
	}

	return input;
}


bool conv2d_transposed::cudnn_status_check(cudnnStatus_t status) const
{
	return status == cudnnStatus_t::CUDNN_STATUS_SUCCESS;
}

void conv2d_transposed::backprop_cudnn(const float* derivative)
{
	float alpha = 1.0f, beta = 0.0f;
	
	cudnnConvolutionBackwardData(handle.handle,
		(void*)(&alpha),
		filter_desc.descriptor, filter_data.get_weights().get(),
		input_desc.descriptor, derivative,
		conv2d_desc.descriptor,
		backprop_algo,
		shared_mem.get(),
		shared_mem.size() * sizeof(float),
		(void*)(&beta),
		out_desc.descriptor, output.get());
}

void conv2d_transposed::backprop_weights_cudnn(const float* derivative, const float* input)
{
	float alpha = 1.0f, beta = 1.0f;
	filter_data.get_weights_derivative().memset(0);
 	cudnnStatus_t status = cudnnConvolutionBackwardFilter(
		handle.handle,
		&alpha,
		out_desc.descriptor,
		derivative,
		input_desc.descriptor,
		input,
		conv2d_desc.descriptor,
		filter_backprop_algo,
		shared_mem.get(),
		shared_mem.size() * sizeof(float),
		&beta,
		filter_desc.descriptor,
		filter_data.get_weights_derivative().get()
	);
	if (status != cudnnStatus_t::CUDNN_STATUS_SUCCESS)
	{
		std::exit(EXIT_FAILURE);
	}
}

void conv2d_transposed::update_weights()
{
	const float alpha = 1.0f, alpha2 = -(learing_rate / output_shape.batches), beta = 0.0f;
	cudnnOpTensor(handle.handle,
		add_tensor.descriptor,
		&alpha,
		weight_desc.descriptor,
		filter_data.get_weights().get(),
		&alpha2,
		weight_desc.descriptor,
		filter_data.get_weights_derivative().get(),
		&beta,
		weight_desc.descriptor,
		filter_data.get_weights().get());
}

weights_properties conv2d_transposed::get_weights() const
{
	weights_properties props;
	props.size = filter_data.get_weights().size();
	props.ptr = filter_data.get_weights().get();
	return props;
}

weights_properties conv2d_transposed::get_weights_deriv() const
{
	weights_properties props;
	props.size = filter_data.get_weights_derivative().size();
	props.ptr = filter_data.get_weights_derivative().get();
	return props;
}