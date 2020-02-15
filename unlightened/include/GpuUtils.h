#pragma once
#ifndef _GPU_UTILS_H
#define _GPU_UTILS_H
#include "cuda_runtime.h"
#include <cmath>
#include <type_traits>
#include <iostream>

namespace utils
{
unsigned int getBlockSize(size_t threadsPerBlock, size_t maxThreads);
bool GpuInit();
void waitAndCheckForErrors();
bool GpuRelase();

template <typename T, std::enable_if_t<std::is_trivially_copyable_v<T>, T>* = nullptr>
class device_struct
{
	T obj;
	T* device_obj;
public:
	device_struct() = delete;

	device_struct(const device_struct& host_s) : obj(host_s.obj)
	{
		copy(obj);
	}

	device_struct(const T& host_obj) : obj(host_obj)
	{
		copy(obj);
	}

	device_struct(device_struct&& host_s) : obj(std::forward(host_s.obj))
	{
		device_obj = host_s.device_obj;
		host_s.device_obj = nullptr;
	}
	
	T* get() const
	{
		return device_obj;
	}

	~device_struct()
	{
		if(device_obj != nullptr)
			cudaFree(device_obj);
	}
private:
	void copy(const T& user_obj)
	{
		if (cudaSuccess != cudaMalloc(&device_obj, sizeof(T)))
		{
			std::cout << "cudaMalloc failed" << std::endl;
		}

		if (cudaSuccess != cudaMemcpy(device_obj ,&user_obj, sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice))
		{
			std::cout << "cudaMemcpy failed" << std::endl;
		}
	}
};
}
#endif
