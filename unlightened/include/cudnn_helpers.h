#pragma once
#include <cudnn.h>
#include <iostream>
#include <shape.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudnnStatus_t code, const char* file, int line, bool abort = true)
{
	if (code != cudnnStatus_t::CUDNN_STATUS_SUCCESS)
	{
		std::cout << "CUDA Error: " << cudnnGetErrorString(code) << " " << file << " " << line;
		if (abort) exit(code);
	}
}

struct cudnn_descriptor4d
{
	shape sh;
	cudnnTensorDescriptor_t descriptor;
	cudnn_descriptor4d(): descriptor(nullptr){}
	cudnn_descriptor4d(const cudnn_descriptor4d& d) = delete;

	cudnn_descriptor4d(cudnn_descriptor4d&& d) noexcept
	{
		descriptor = d.descriptor;
		d.descriptor = nullptr;
	}

	bool create(int width, int height, int depth, int batches)
	{
		sh.width = width;
		sh.height = height;
		sh.depth = depth;
		sh.batches = batches;
		cudnnStatus_t status = cudnnCreateTensorDescriptor(&descriptor);
		if (status != cudnnStatus_t::CUDNN_STATUS_SUCCESS)
			return false;

		status = cudnnSetTensor4dDescriptor(descriptor,
			/*format=*/CUDNN_TENSOR_NCHW, // row major
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*batch_size=*/batches,
			/*channels=*/depth,
			/*image_height=*/height,
			/*image_width=*/width);

		return status == cudnnStatus_t::CUDNN_STATUS_SUCCESS;
	}


	~cudnn_descriptor4d()
	{
		if (descriptor != nullptr)
		{
			cudnnDestroyTensorDescriptor(descriptor);
		}
	}
};

struct cudnn_filter_descriptor
{
	cudnnFilterDescriptor_t descriptor = nullptr;
	cudnn_filter_descriptor() : descriptor(nullptr) {}
	cudnn_filter_descriptor(const cudnn_filter_descriptor& d) = delete;

	cudnn_filter_descriptor(cudnn_filter_descriptor&& d) noexcept
	{
		descriptor = d.descriptor;
		d.descriptor = nullptr;
	}

	bool create(int width, int height, int channels, int num_of_filters)
	{
		cudnnStatus_t status = cudnnCreateFilterDescriptor(&descriptor);
		if (status != cudnnStatus_t::CUDNN_STATUS_SUCCESS)
			return false;

		status = cudnnSetFilter4dDescriptor(descriptor,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*out_channels=*/num_of_filters,
			/*in_channels=*/channels,
			/*kernel_height=*/height,
			/*kernel_width=*/width);

		return status == cudnnStatus_t::CUDNN_STATUS_SUCCESS;
	}

	~cudnn_filter_descriptor()
	{
		if (descriptor != nullptr)
		{
			cudnnDestroyFilterDescriptor(descriptor);
		}
	}
}; 

struct cudnn_conv2d_descriptor
{
	cudnnConvolutionDescriptor_t descriptor = nullptr;
	cudnn_conv2d_descriptor() : descriptor(nullptr) {}
	cudnn_conv2d_descriptor(const cudnn_conv2d_descriptor& d) = delete;

	cudnn_conv2d_descriptor(cudnn_conv2d_descriptor&& d) noexcept
	{
		descriptor = d.descriptor;
		d.descriptor = nullptr;
	}

	bool create(int padding_w, int padding_h, int stride)
	{
		cudnnStatus_t status = cudnnCreateConvolutionDescriptor(&descriptor);
		if (status != cudnnStatus_t::CUDNN_STATUS_SUCCESS)
			return false;

		status = cudnnSetConvolution2dDescriptor(descriptor,
			/*pad_height=*/padding_h,
			/*pad_width=*/padding_w,
			/*vertical_stride=*/stride,
			/*horizontal_stride=*/stride,
			/*dilation_height=*/1,
			/*dilation_width=*/1,
			/*mode=*/CUDNN_CROSS_CORRELATION, // check this
			/*computeType=*/CUDNN_DATA_FLOAT);

		return status == cudnnStatus_t::CUDNN_STATUS_SUCCESS;
	}

	~cudnn_conv2d_descriptor()
	{
		if (descriptor != nullptr)
		{
			cudnnDestroyConvolutionDescriptor(descriptor);
		}
	}
};	// create add op tensor 


struct cudnn_add_tensor
{
	cudnnOpTensorDescriptor_t descriptor = nullptr;
	cudnn_add_tensor() : descriptor(nullptr) {}
	cudnn_add_tensor(const cudnn_add_tensor& d) = delete;

	cudnn_add_tensor(cudnn_add_tensor&& d) noexcept
	{
		descriptor = d.descriptor;
		d.descriptor = nullptr;
	}

	bool create()
	{
		cudnnStatus_t status = cudnnCreateOpTensorDescriptor(&descriptor);
		if (status != cudnnStatus_t::CUDNN_STATUS_SUCCESS)
			return false;

		status = cudnnSetOpTensorDescriptor(descriptor,
			CUDNN_OP_TENSOR_ADD,
			CUDNN_DATA_FLOAT,
			CUDNN_NOT_PROPAGATE_NAN);

		return status == cudnnStatus_t::CUDNN_STATUS_SUCCESS;
	}

	~cudnn_add_tensor()
	{
		if (descriptor != nullptr)
		{
			cudnnDestroyOpTensorDescriptor(descriptor);
		}
	}
};

struct cudnn_hanlde
{
	cudnnHandle_t handle;
	cudnn_hanlde()
	{
		if (cudnnStatus_t::CUDNN_STATUS_SUCCESS != cudnnCreate(&handle))
			handle = nullptr;
	}

	cudnn_hanlde(const cudnn_hanlde& d) = delete;

	cudnn_hanlde(cudnn_hanlde&& d) noexcept
	{
		handle = d.handle;
		d.handle = nullptr;
	}

	operator bool() const { return handle != nullptr; }

	~cudnn_hanlde()
	{
		if (handle != nullptr)
		{
			cudnnDestroy(handle);
		}
	}
};