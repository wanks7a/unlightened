#include <conv2d_cudnn.h>

void conv2d_cudnn::checkCUDNN(const cudnnStatus_t& status)
{
	if (status != cudnnStatus_t::CUDNN_STATUS_SUCCESS)
	{
		std::cerr << "Error on line " << __LINE__ << ": "
			<< cudnnGetErrorString(status) << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

conv2d_cudnn::conv2d_cudnn() : is_first_layer(false)
{
	in_device_memory = true;
	checkCUDNN(cudnnCreate(&cudnn_handle));
}

conv2d_cudnn::conv2d_cudnn(size_t filter_dimension, size_t num_of_filters, bool first_layer) : options(filter_dimension, filter_dimension, num_of_filters), filters_size(num_of_filters), input_layer(nullptr), is_first_layer(first_layer)
{
	in_device_memory = true;
	checkCUDNN(cudnnCreate(&cudnn_handle));
}

conv2d_cudnn::conv2d_cudnn(const filter_options& opt, bool first_layer) : filters_size(opt.num_of_filters), input_layer(nullptr), is_first_layer(first_layer)
{
	in_device_memory = true;
	options = opt;
	checkCUDNN(cudnnCreate(&cudnn_handle));
}

void conv2d_cudnn::init(const shape& input)
{
	options.channels = input.depth;
	filters.set_options(options);
	filters.init(input);
	output_shape = filters.get_output_shape();
	output.resize(output_shape.size());
	input_derivative.resize(input.size());
	filters.get_weights_derivative().resize(filters.get_filter_shape().size() * filters.get_options().num_of_filters);
	init_cudnn();
}

const float* conv2d_cudnn::get_output() const
{
	return output.get();
}

const float* conv2d_cudnn::derivative_wr_to_input() const
{
	return input_derivative.get();
}

void conv2d_cudnn::init_cudnn()
{
	initialized = true;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
		/*format=*/CUDNN_TENSOR_NCHW, // row major
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/input_shape.batches,
		/*channels=*/input_shape.depth,
		/*image_height=*/input_shape.height,
		/*image_width=*/input_shape.width));
	checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/output_shape.batches,
		/*channels=*/output_shape.depth,
		/*image_height=*/output_shape.height,
		/*image_width=*/output_shape.width));
	checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
	checkCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*out_channels=*/output_shape.depth,
		/*in_channels=*/filters.get_options().channels,
		/*kernel_height=*/filters.get_options().h,
		/*kernel_width=*/filters.get_options().w));

	checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_forwardpass_descriptor));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_forwardpass_descriptor,
		/*pad_height=*/filters.get_padding(),
		/*pad_width=*/filters.get_padding(),
		/*vertical_stride=*/options.stride,
		/*horizontal_stride=*/options.stride,
		/*dilation_height=*/1,
		/*dilation_width=*/1,
		/*mode=*/CUDNN_CROSS_CORRELATION, // check this
		/*computeType=*/CUDNN_DATA_FLOAT));
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handle,
		input_descriptor,
		filter_descriptor,
		convolution_forwardpass_descriptor,
		output_descriptor,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		/*memoryLimitInBytes=*/0,
		&convolution_forwardpass_algorithm));
	size_t workspace_bytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
		input_descriptor,
		filter_descriptor,
		convolution_forwardpass_descriptor,
		output_descriptor,
		convolution_forwardpass_algorithm,
		&workspace_bytes));

	if ((workspace_bytes % sizeof(float)) != 0)
	{
		std::cout << "error requested memory from cudnn is not divisiable by 4" << std::endl;
	}
	else
	{
		cudnn_memory_forward_pass.resize(workspace_bytes / sizeof(float));
	}

	workspace_bytes = 0;
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle,
		filter_descriptor,
		output_descriptor,
		convolution_forwardpass_descriptor,
		input_descriptor,
		backprop_algo,
		&workspace_bytes));

	if ((workspace_bytes % sizeof(float)) != 0)
	{
		std::cout << "error requested memory from cudnn is not divisiable by 4" << std::endl;
	}
	else
	{
		cudnn_memory_backprop.resize(workspace_bytes / sizeof(float));
	}

	workspace_bytes = 0;
	cudnnConvolutionBwdFilterPreference_t algo_pref = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle,
		input_descriptor,
		output_descriptor,
		convolution_forwardpass_descriptor,
		filter_descriptor,
		algo_pref,
		0,
		&filter_backprop_algo));
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle,
		input_descriptor,
		output_descriptor,
		convolution_forwardpass_descriptor,
		filter_descriptor,
		filter_backprop_algo,
		&workspace_bytes
	));

	if ((workspace_bytes % sizeof(float)) != 0)
	{
		std::cout << "error requested memory from cudnn is not divisiable by 4" << std::endl;
	}
	else
	{
		cudnn_memory_backprop_filter.resize(workspace_bytes / sizeof(float));
	}

	// create add op tensor 
	checkCUDNN(cudnnCreateOpTensorDescriptor(&add_op_descriptor));
	checkCUDNN(cudnnSetOpTensorDescriptor(add_op_descriptor,
		CUDNN_OP_TENSOR_ADD,
		CUDNN_DATA_FLOAT,
		CUDNN_PROPAGATE_NAN));

	checkCUDNN(cudnnCreateTensorDescriptor(&bias_tensor_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(bias_tensor_descriptor,
		/*format=*/CUDNN_TENSOR_NCHW, // row major
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/1,
		/*channels=*/filters.get_options().num_of_filters,
		/*image_height=*/1,
		/*image_width=*/1));

	checkCUDNN(cudnnCreateTensorDescriptor(&weights_tensor_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(weights_tensor_descriptor,
		/*format=*/CUDNN_TENSOR_NCHW, // row major
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/filters.get_options().num_of_filters,
		/*channels=*/filters.get_options().channels,
		/*image_height=*/filters.get_options().h,
		/*image_width=*/filters.get_options().w));
}

conv2d_cudnn::~conv2d_cudnn()
{
	if (initialized)
	{
		cudnnDestroyOpTensorDescriptor(add_op_descriptor);
		cudnnDestroyTensorDescriptor(bias_tensor_descriptor);
		cudnnDestroyTensorDescriptor(input_descriptor);
		cudnnDestroyTensorDescriptor(output_descriptor);
		cudnnDestroyFilterDescriptor(filter_descriptor);
		cudnnDestroyConvolutionDescriptor(convolution_forwardpass_descriptor);
	}
	cudnnDestroy(cudnn_handle);
}

void conv2d_cudnn::backprop_cudnn(const float* derivative)
{
	float alpha = 1.0f, beta = 0.0f;

	checkCUDNN(cudnnConvolutionBackwardData(cudnn_handle,
		(void*)(&alpha),
		filter_descriptor, filters.get_weights().get(),
		output_descriptor, derivative,
		convolution_forwardpass_descriptor,
		backprop_algo,
		cudnn_memory_backprop.get(),
		cudnn_memory_backprop.size() * sizeof(float),
		(void*)(&beta),
		input_descriptor, input_derivative.get()));
}

void conv2d_cudnn::backprop_weights_cudnn(const float* derivative)
{
	float alpha = 1.0f, beta = 1.0f;
	filters.get_weights_derivative().memset(0);
	checkCUDNN(cudnnConvolutionBackwardFilter(
		cudnn_handle,
		&alpha,
		input_descriptor,
		input,
		output_descriptor,
		derivative,
		convolution_forwardpass_descriptor,
		filter_backprop_algo,
		cudnn_memory_backprop_filter.get(),
		cudnn_memory_backprop_filter.size() * sizeof(float),
		&beta,
		filter_descriptor,
		filters.get_weights_derivative().get()
	));
}

void conv2d_cudnn::update_weights()
{
	const float alpha = 1.0f, alpha2 = -(learing_rate / output_shape.batches), beta = 0.0f;
	checkCUDNN(cudnnOpTensor(cudnn_handle,
		add_op_descriptor,
		&alpha,
		weights_tensor_descriptor,
		filters.get_weights().get(),
		&alpha2,
		weights_tensor_descriptor,
		filters.get_weights_derivative().get(),
		&beta,
		weights_tensor_descriptor,
		filters.get_weights().get()));
}

weights_properties conv2d_cudnn::get_weights() const 
{
	weights_properties props;
	props.size = filters.get_weights().size();
	props.ptr = filters.get_weights().get();
	return props;
};

weights_properties conv2d_cudnn::get_weights_deriv() const
{
	weights_properties props;
	props.size = filters.get_weights_derivative().size();
	props.ptr = filters.get_weights_derivative().get();
	return props;
};

weights_properties conv2d_cudnn::get_bias() const
{
	weights_properties props;
	props.size = filters.get_bias().size();
	props.ptr = filters.get_bias().get();
	return props;
};

weights_properties conv2d_cudnn::get_bias_deriv() const
{
	weights_properties props;
	props.size = filters.get_bias_derivative().size();
	props.ptr = filters.get_bias_derivative().get();
	return props;
};

bool conv2d_cudnn::forward(std::shared_ptr<cuda_device>& d, const blob_view<float>& input_data)
{
	const float alpha = 1.0f, alpha2 = 1.0f, beta = 0.0f;
	input = input_data.data();
	// convolution 
	checkCUDNN(cudnnConvolutionForward(cudnn_handle,
		&alpha,
		input_descriptor,
		input,
		filter_descriptor,
		filters.get_weights().get(),
		convolution_forwardpass_descriptor,
		convolution_forwardpass_algorithm,
		cudnn_memory_forward_pass.get(),
		cudnn_memory_forward_pass.size() * sizeof(float),
		&beta,
		output_descriptor,
		output.get()));

	// add bias to the output from the convolution
	checkCUDNN(cudnnOpTensor(cudnn_handle,
		add_op_descriptor,
		&alpha,
		output_descriptor,
		output.get(),
		&alpha2,
		bias_tensor_descriptor,
		filters.get_bias().get(),
		&beta,
		output_descriptor,
		output.get()));
	return true;
}

bool conv2d_cudnn::backward(std::shared_ptr<cuda_device>& d, const blob_view<float>& derivative_data)
{
	backprop_weights_cudnn(derivative_data.data());
	// compute bias gradient
	float alpha = 1.0f, beta = 0.0f;
	checkCUDNN(cudnnConvolutionBackwardBias(cudnn_handle,
		&alpha,
		output_descriptor,
		derivative_data.data(),
		&beta,
		bias_tensor_descriptor,
		filters.get_bias_derivative().get()
	));

	backprop_cudnn(derivative_data.data());

	return true;
}