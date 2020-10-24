#pragma once
#include <Layer.h>

template <typename T>
class serializable_layer : public Layer
{
public:
	void serialize(binary_serialization& s) const final
	{
		s.serialize(static_cast<const T&>(*this));
		Layer::serialize(s);
		const T& ref = static_cast<const T&>(*this);
		ref.serialize_members(s);
	}

	bool deserialize(binary_serialization& s) final
	{
		if (s.deserialize(static_cast<T&>(*this)))
		{
			Layer::deserialize(s);
			T& ref = static_cast<T&>(*this);
			ref.deserialize_members(s);
			return true;
		}
		return false;
	}

private:
	serializable_layer() = default;
	friend T;
};