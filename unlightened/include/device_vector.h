#pragma once
template <typename Device, typename DType>
class device_vector
{
	DType* mem;
	size_t data_size;

public:
	device_vector() : mem(nullptr), data_size(0)
	{
	}

	device_vector(device_vector&& v) noexcept
	{
		mem = v.mem;
		data_size = v.data_size;
		v.mem = nullptr;
		v.data_size = 0;
	}

	void reserve(size_t req_size)
	{
		if (req_size != data_size)
		{
			DType* req_mem = Device::template malloc<DType>(req_size);
			clear();
			mem = req_mem;
			data_size = req_size;
		}
	}

	void resize(size_t req_size, const DType& val)
	{
		std::vector<DType> values;
		values.resize(req_size, val);
		set_data(values);
	}

	void set_data(const DType* d, size_t size)
	{
		reserve(size);
		Device::copy_to_device(this->mem, d, size);
	}

	void set_data(const std::vector<DType>& d)
	{
		set_data(d.data(), d.size());
	}

	std::vector<DType> to_vector() const
	{
		std::vector<DType> result;
		result.resize(data_size);
		Device::copy_to_host(result.data(), mem, data_size);
		return result;
	}

	device_vector copy() const
	{
		device_vector result;
		result.mem = Device:: template malloc<DType>(data_size);
		result.data_size = data_size;
		Device::memcpy(result.mem, mem, data_size);
		return result;
	}

	DType* data()
	{
		return mem;
	}

	DType* data() const
	{
		return mem;
	}
	
	size_t size() const
	{
		return data_size;
	}

	void clear()
	{
		if (mem != nullptr)
		{
			Device::free(mem);
			data_size = 0;
		}
	}

	~device_vector()
	{
		clear();
	}
};