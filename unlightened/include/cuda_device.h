#pragma once
#include "cuda_runtime.h"

struct cuda_device
{
	template <typename DType>
	static DType* malloc(size_t size)
	{
		DType* ptr = nullptr;
		if (cudaSuccess != cudaMalloc(&ptr, sizeof(DType) * size))
		{
			std::cout << "cudaMalloc failed" << std::endl;
		}
		return ptr;
	}

	template <typename DType>
	static void free(DType* ptr)
	{
		if (cudaSuccess != cudaFree(ptr))
		{
			std::cout << "cudaFree failed" << std::endl;
		}
	}

	template <typename DType>
	static void memcpy(DType* dest, const DType* src, size_t size)
	{
		if (cudaSuccess != cudaMemcpy(dest, src, sizeof(DType) * size, cudaMemcpyKind::cudaMemcpyDeviceToDevice))
		{
			std::cout << "cudaMemcpy failed" << std::endl;
		}
	}

	template <typename DType>
	static void copy_to_host(DType* dest, const DType* src, size_t size)
	{
		if (cudaSuccess != cudaMemcpy(dest, src, sizeof(DType) * size, cudaMemcpyKind::cudaMemcpyDeviceToHost))
		{
			std::cout << "cudaMemcpy failed" << std::endl;
		}
	}

	template <typename DType>
	static void copy_to_device(DType* dest, const DType* src, size_t size)
	{
		if (cudaSuccess != cudaMemcpy(dest, src, sizeof(DType) * size, cudaMemcpyKind::cudaMemcpyHostToDevice))
		{
			std::cout << "cudaMemcpy failed" << std::endl;
		}
	}

	template <typename DType>
	static void memset(DType* data, size_t size, unsigned char value)
	{
		if (cudaSuccess != cudaMemset(data, value, sizeof(DType) * size))
		{
			std::cout << "cudaMemcpy failed" << std::endl;
		}
	}
};