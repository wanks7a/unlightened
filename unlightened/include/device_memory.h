#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>

template <typename T>
class cuVector
{
	size_t currentSize = 0;
	T* ptr;
public:
	cuVector() : ptr(nullptr) {};
	
	cuVector(const cuVector<T>& v) = delete;

	cuVector(cuVector<T>&& v) noexcept
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

	static std::vector<T> from_device_host(const T* ptr, size_t size)
	{
		std::vector<T> result;
		result.resize(size);
		if (cudaSuccess != cudaMemcpy(result.data(), ptr, sizeof(T) * size, cudaMemcpyKind::cudaMemcpyDeviceToHost))
		{
			std::cout << "cudaMemcpy failed" << std::endl;
		}
		return result;
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

	bool resize(size_t size)
	{
		if (currentSize != size)
		{
			dealloc();
			currentSize = size;
			if (cudaSuccess != cudaMalloc(&ptr, sizeof(T) * size))
			{
				std::cout << "cudaMalloc failed" << std::endl;
				return false;
			}
			if (cudaSuccess != cudaMemset(ptr, 0, sizeof(T) * size))
			{
				std::cout << "failed to zero out the new memory" << std::endl;
			}
		}
		return true;
	}

	bool resize(size_t size,const T& value)
	{
		if (currentSize != size)
		{
			dealloc();
			currentSize = size;
			std::vector<T> values;
			values.resize(size, value);
			if (cudaSuccess != cudaMalloc(&ptr, sizeof(T) * size))
			{
				std::cout << "cudaMalloc failed" << std::endl;
				return false;
			}
			if (cudaSuccess != cudaMemcpy(ptr, values.data(), sizeof(T) * size, cudaMemcpyKind::cudaMemcpyHostToDevice))
			{
				std::cout << "cudaMemcpy failed" << std::endl;
			}
		}
		return true;
	}

	size_t size()
	{
		return currentSize;
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