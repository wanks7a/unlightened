#pragma once
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <generic_stream.h>
#include <cstdint>

class Layer;
class dense_layer;

class binary_serialization 
{
	std::shared_ptr<generic_stream> stream;
	std::unordered_map<int, Layer* (binary_serialization::*)()> default_constructors;

	enum TYPE_ID
	{
		DENSE = 1,
		DENSE_GPU = 2,
		CONV_2D = 3,
		CONV_2D_GPU = 4
	};

	template <typename T>
	Layer* construct() { return new T(); }

	template <typename T>
	void register_def_constructor(int id)
	{
		default_constructors[id] = &binary_serialization::construct<T>;
	}
public:
	binary_serialization(std::shared_ptr<generic_stream> s);
	void serialize(const Layer& obj);
	std::shared_ptr<Layer> deserialize_layer();
	void serialize(const dense_layer& l);
	bool binary_serialization::deserialize(dense_layer& l);

	template <typename T, typename = std::enable_if<std::is_trivially_copyable<T>::value>::type>
	binary_serialization& operator<<(const T& value)
	{
		stream->write((const char*)(&value), sizeof(T));
		return *this;
	}

	template <typename T>
	binary_serialization& operator<<(const std::vector<T>& values)
	{
		*this << static_cast<uint32_t>(values.size());
		for (const auto& v : values)
		{
			*this << v;
		}
		return *this;
	}

	template <typename T, typename = std::enable_if<std::is_trivially_copyable<T>::value>::type>
	binary_serialization& operator>>(T& value)
	{
		if (stream->read((char*)(&value), sizeof(T)) != sizeof(T))
		{
			throw("size missmatch");
		}
		return *this;
	}

	template <typename T>
	binary_serialization& operator>>(std::vector<T>& values)
	{
		uint32_t size = 0;
		*this >> size;
		values.reserve(size);
		for (size_t i = 0; i < size; i++)
		{
			T value;
			*this >> value;
			values.emplace_back(value);
		}
		return *this;
	}

	template <>
	binary_serialization& operator>><float>(std::vector<float>& values)
	{
		uint32_t size = 0;
		*this >> size;
		values.resize(size);
		stream->read((char*)values.data(), size * sizeof(float));
		return *this;
	}
};