#pragma once
#include <device_vector.h>

template <typename DType>
class blob_view
{
	const DType* ptr;
	size_t count;
public:

	blob_view(DType* ptr = nullptr, size_t size = 0) : ptr(ptr), count(size) {};

	void set_size(const DType* ptr, size_t size)
	{
		this->ptr = ptr;
		count = size;
	}

	size_t size() const
	{
		return count;
	}

	const DType* data() const
	{
		return ptr;
	}

	template <typename Device>
	device_vector<Device, DType> to_device(Device& d) const
	{
		device_vector<Device, DType> result;
		result.memcpy(ptr, count);
		return result;
	}
};