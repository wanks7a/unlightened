#include <binary_serialization.h>
#include <Layer.h>
#include <LinearLayer.h>

binary_serialization::binary_serialization(std::shared_ptr<generic_stream> s) : stream(s)
{
	register_def_constructor<dense_layer>(TYPE_ID::DENSE);
}

void binary_serialization::serialize(const dense_layer& l)
{
	*this << static_cast<uint32_t>(TYPE_ID::DENSE);
}

bool binary_serialization::deserialize(dense_layer& l)
{
	uint32_t layer_id;
	*this >> layer_id;
	return layer_id == TYPE_ID::DENSE;
}

void binary_serialization::serialize(const Layer& obj)
{
	obj.serialize(*this);
}

std::shared_ptr<Layer> binary_serialization::deserialize_layer()
{
	uint32_t layer_id;
	stream->peek((char*)&layer_id, sizeof(uint32_t));
	auto it = default_constructors.find(layer_id);
	if (it != default_constructors.end())
	{
		std::shared_ptr<Layer> result((this->*it->second)());
		result->deserialize(*this);
		return result;
	}
	return nullptr;
}