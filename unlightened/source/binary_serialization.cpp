#include <binary_serialization.h>
#include <Layer.h>
#include <LinearLayer.h>
#include <LinearLayerGPU.h>
#include <activation_layer.h>
#include <max_pool.h>

template <typename T>
void binary_serialization::serialize()
{
	*this << static_cast<uint32_t>(is_layer<T>::id);
}

template <typename T>
bool binary_serialization::deserialize()
{
	uint32_t layer_id;
	*this >> layer_id;
	return layer_id == static_cast<uint32_t>(is_layer<T>::id);
}

template <typename T>
struct binary_serialization::is_layer
{
};

#define LAYER_MAP(NAME, ID)     template <> \
								struct binary_serialization::is_layer<NAME> \
								{ static constexpr int id = ID; }; \
								template void binary_serialization::serialize<NAME>(); \
								template bool binary_serialization::deserialize<NAME>();

LAYER_MAP(dense_layer,			binary_serialization::TYPE_ID::DENSE);
LAYER_MAP(dense_gpu,			binary_serialization::TYPE_ID::DENSE_GPU);
LAYER_MAP(activation_layer,		binary_serialization::TYPE_ID::ACTIVATION);

binary_serialization::binary_serialization(std::shared_ptr<generic_stream> s) : stream(s)
{
	register_def_constructor<dense_layer>(TYPE_ID::DENSE);
	register_def_constructor<dense_gpu>(TYPE_ID::DENSE_GPU);
	register_def_constructor<activation_layer>(TYPE_ID::ACTIVATION);
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
