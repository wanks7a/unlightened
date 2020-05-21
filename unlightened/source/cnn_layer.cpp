#include <cnn_layer.h>

void cnn_layer::checkCUDNN(const cudnnStatus_t& status)
{                                                                             
	if (status != CUDNN_STATUS_SUCCESS) 
	{
		std::cerr << "Error on line " << __LINE__ << ": "      
		<< cudnnGetErrorString(status) << std::endl; 
		std::exit(EXIT_FAILURE);                               
	}                                                        
}

cnn_layer::cnn_layer(size_t filter_dimension, size_t num_of_filters, bool first_layer) : options(filter_dimension, filter_dimension, num_of_filters), filters_size(num_of_filters), input_layer(nullptr), is_first_layer(first_layer)
{
	device_layer = true;
	if(use_cudnn)
		checkCUDNN(cudnnCreate(&cudnn_handle));
}

void cnn_layer::init(const shape& input)
{
	options.channels = input.depth;
	filters.set_options(options);
	filters.init(input);	
	output_shape = filters.get_output_shape();
	output.resize(output_shape.size());
	input_derivative.resize(input.size());
	if (use_cudnn)
		init_cudnn();
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

	if (use_cudnn)
	{
		const float alpha = 1.0f, beta = 0.0f;
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
		add_bias_to_output(output, output_shape, filters.get_bias());
	}
	else
	{
		for (int i = 0; i < filters.size(); i++)
		{
			conv_3d(input, input_layer->get_shape(), output.get() + i * output_shape.area(), output_shape, filters[i], filters.get_bias().get() + i, options.w, options.h, options.w - filters.get_padding());
		}
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

	if (!is_first_layer)
	{
		if (use_cudnn)
		{
			backprop_cudnn(derivative);
		}
		else
		{
			derivative_input_3d(weights_flipped.get(), filter_shape, input_derivative.get(), input_shape,
				derivative, output_shape.width, output_shape.height, output_shape.width - filters.get_padding(), output_shape.volume());
		}
	}
	if (update_on_backprop)
	{
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
}

const float* cnn_layer::get_output()
{
	return output.get();
}

const float* cnn_layer::derivative_wr_to_input()
{
	return input_derivative.get();
}

void cnn_layer::init_cudnn()
{
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
		/*vertical_stride=*/1,
		/*horizontal_stride=*/1,
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
}

cnn_layer::~cnn_layer()
{
	if (use_cudnn)
	{
		cudnnDestroyTensorDescriptor(input_descriptor);
		cudnnDestroyTensorDescriptor(output_descriptor);
		cudnnDestroyFilterDescriptor(filter_descriptor);
		cudnnDestroyConvolutionDescriptor(convolution_forwardpass_descriptor);
		cudnnDestroy(cudnn_handle);
	}
}

void cnn_layer::backprop_cudnn(const float* derivative)
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