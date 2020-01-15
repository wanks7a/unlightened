#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

template <typename T>
class cuVector
{
	size_t currentSize = 0;
	T* ptr;
public:
	cuVector() : ptr(nullptr) {};
	
	cuVector(const cuVector<T>& v) = delete;

	cuVector(cuVector<T>&& v)
	{
		currentSize = v.currentSize;
		ptr = v.ptr;
		v.ptr = nullptr;
		v.currentSize = 0;
	}

	T* get() const
	{
		return ptr;
	}

	void getCopy(std::vector<T>& obj)
	{
		if (obj.size() != currentSize)
		{
			obj.resize(currentSize);
		}
		if (cudaSuccess != cudaMemcpy(obj.data(), ptr, sizeof(T) * currentSize, cudaMemcpyKind::cudaMemcpyDeviceToHost))
		{
			std::cout << "cudaMemcpy failed" << std::endl;
		}
	}

	bool setValues(const T* values, size_t count)
	{
		if (currentSize != count || ptr == nullptr)
		{
			dealloc();
			currentSize = count;
			if (cudaSuccess != cudaMalloc(&ptr, sizeof(T) * count))
			{
				std::cout << "cudaMalloc failed" << std::endl;
			}
		}
		if (cudaSuccess != cudaMemcpy(ptr, values, sizeof(T) * count, cudaMemcpyKind::cudaMemcpyHostToDevice))
		{
			std::cout << "cudaMemcpy failed" << std::endl;
		}
		return true;
	}

	bool setValues(const std::vector<T>& values)
	{
		return setValues(values.data(), values.size());
	}

	~cuVector()
	{
		dealloc();
	}

private:
	void dealloc()
	{
		if (ptr != nullptr)
		{
			cudaFree(ptr);
		}
		currentSize = 0;
	}
};