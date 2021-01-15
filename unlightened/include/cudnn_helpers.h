#pragma once
#include <cudnn.h>
#include <iostream>
#include <shape.h>
#include <device_vector.h>
#include <cuda_device.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudnnStatus_t code, const char* file, int line, bool abort = true)
{
	if (code != cudnnStatus_t::CUDNN_STATUS_SUCCESS)
	{
		std::cout << "CUDA Error: " << cudnnGetErrorString(code) << " " << file << " " << line;
		if (abort) exit(code);
	}
}

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
		clear();
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
		CUDA_CHECK(status);

		return status == cudnnStatus_t::CUDNN_STATUS_SUCCESS;
	}

	bool create(int size)
	{
		return create(size, 1, 1, 1);
	}

	void scale(float* ptr, float scaleVal)
	{
		if (descriptor)
		{
			cudnn_hanlde handle;
			CUDA_CHECK(cudnnScaleTensor(handle.handle, descriptor, ptr, &scaleVal));
		}
	}

	~cudnn_descriptor4d()
	{
		clear();
	}

private:
	void clear()
	{
		if (descriptor != nullptr)
		{
			cudnnDestroyTensorDescriptor(descriptor);
			descriptor = nullptr;
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
		clear();
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
		clear();
	}
private:
	void clear()
	{
		if (descriptor != nullptr)
		{
			cudnnDestroyFilterDescriptor(descriptor);
			descriptor = nullptr;
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
		clear();
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
		clear();
	}
private:
	void clear()
	{
		if (descriptor != nullptr)
		{
			cudnnDestroyConvolutionDescriptor(descriptor);
			descriptor = nullptr;
		}
	}
};

template <unsigned int Type>
struct cudnn_op_tensor
{
	cudnnOpTensorDescriptor_t descriptor = nullptr;
	cudnn_op_tensor() : descriptor(nullptr) {}
	cudnn_op_tensor(const cudnn_op_tensor& d) = delete;

	cudnn_op_tensor(cudnn_op_tensor&& d) noexcept
	{
		descriptor = d.descriptor;
		d.descriptor = nullptr;
	}

	bool create()
	{
		clear();
		cudnnStatus_t status = cudnnCreateOpTensorDescriptor(&descriptor);
		if (status != cudnnStatus_t::CUDNN_STATUS_SUCCESS)
			return false;

		status = cudnnSetOpTensorDescriptor(descriptor,
			static_cast<cudnnOpTensorOp_t>(Type),
			CUDNN_DATA_FLOAT,
			CUDNN_PROPAGATE_NAN);
		CUDA_CHECK(status);
		return status == cudnnStatus_t::CUDNN_STATUS_SUCCESS;
	}

	bool op(const float* A, size_t A_size, const float* B, size_t B_size, float* C, size_t C_size, float alpha1 = 1.0f, float alpha2 = 1.0f, float beta = 0.0f)
	{
		cudnn_hanlde handle;
		cudnn_descriptor4d A_desc, B_desc, C_desc;
		if (!A_desc.create(A_size) || !B_desc.create(B_size) || !C_desc.create(C_size))
			return false;
		CUDA_CHECK(cudnnOpTensor(handle.handle,
			descriptor,
			&alpha1,
			A_desc.descriptor,
			A,
			&alpha2,
			B_desc.descriptor,
			B,
			&beta,
			C_desc.descriptor,
			C));
		return true;
	}

	~cudnn_op_tensor()
	{
		clear();
	}
private:
	void clear()
	{
		if (descriptor != nullptr)
		{
			cudnnDestroyOpTensorDescriptor(descriptor);
			descriptor = nullptr;
		}
	}
};

using cudnn_add_tensor = cudnn_op_tensor<CUDNN_OP_TENSOR_ADD>;
using cudnn_mul_tensor = cudnn_op_tensor<CUDNN_OP_TENSOR_MUL>;
using cudnn_sqrt_tensor = cudnn_op_tensor<CUDNN_OP_TENSOR_SQRT>;

template <unsigned int Type>
struct cudnn_reduce_op_tensor
{
	cudnnReduceTensorDescriptor_t descriptor = nullptr;
	cudnn_reduce_op_tensor() : descriptor(nullptr), workspaceBytes(0) {}
	cudnn_reduce_op_tensor(const cudnn_reduce_op_tensor& d) = delete;

	cudnn_reduce_op_tensor(cudnn_reduce_op_tensor&& d) noexcept
	{
		descriptor = d.descriptor;
		workspaceBytes = d.workspaceBytes;
		d.descriptor = nullptr;
		d.indicesBytes = 0;
	}

	bool create(shape input, shape output)
	{
		clear();
		cudnnStatus_t status = cudnnCreateReduceTensorDescriptor(&descriptor);
		if (status != cudnnStatus_t::CUDNN_STATUS_SUCCESS)
			return false;

		status = cudnnSetReduceTensorDescriptor(descriptor,
			static_cast<cudnnReduceTensorOp_t>(Type),
			CUDNN_DATA_FLOAT,
			CUDNN_PROPAGATE_NAN,
			CUDNN_REDUCE_TENSOR_NO_INDICES,
			CUDNN_32BIT_INDICES);

		CUDA_CHECK(status);
		cudnn_hanlde handle;
		if (!input_desc.create(input.width, input.height, input.depth, input.batches))
			return false;
		if (!output_desc.create(output.width, output.height, output.depth, output.batches))
			return false;
		CUDA_CHECK(cudnnGetReductionWorkspaceSize(handle.handle,
			descriptor,
			input_desc.descriptor,
			output_desc.descriptor,
			&workspaceBytes));

		return status == cudnnStatus_t::CUDNN_STATUS_SUCCESS;
	}

	bool create(int reduce_size)
	{
		return create(shape(reduce_size), shape(1));
	}

	bool reduce(const float* input, float* output, float alpha = 1.0f, float beta = 0.0f)
	{
		if (input == nullptr || output == nullptr)
			return false;
		cudnn_hanlde handle;
		device_vector<cuda_device, char> rBytes;
		rBytes.reserve(workspaceBytes);
		CUDA_CHECK(cudnnReduceTensor(handle.handle,
			descriptor,
			nullptr,
			0,
			rBytes.data(),
			workspaceBytes,
			&alpha,
			input_desc.descriptor,
			input,
			&beta,
			output_desc.descriptor,
			output));
		return true;
	}

	~cudnn_reduce_op_tensor()
	{
		clear();
	}
private:
	void clear()
	{
		if (descriptor != nullptr)
		{
			cudnnDestroyReduceTensorDescriptor(descriptor);
			descriptor = nullptr;
		}
	}

	cudnn_descriptor4d input_desc;
	cudnn_descriptor4d output_desc;
	size_t workspaceBytes;
};

using cudnn_reduce_add_tensor = cudnn_reduce_op_tensor<CUDNN_REDUCE_TENSOR_ADD>;
using cudnn_reduce_mul_tensor = cudnn_reduce_op_tensor<CUDNN_REDUCE_TENSOR_MUL>;

bool cuda_scale_vector(float* ptr, size_t size, float scale);
