#pragma once
#include <device_layer.h>
#include <serializable_interface.h>

template <typename Device>
class batch_norm : public  device_layer<batch_norm<Device>, Device, float, true>
{
    template <typename Serializer>
    void serialize_weights_props(Serializer& s, const weights_properties& p) const
    {
        std::vector<float> data;
        data.resize(p.size);
        Device::copy_to_host(data.data(), p.ptr, p.size);
        s << data;
    }

    template <typename Serializer>
    bool deserialize_weights_props(Serializer& s, const weights_properties& p) const
    {
        std::vector<float> data;
        s >> data;
        if (data.size() != p.size)
            return false;
        Device::copy_to_device(p.ptr, data.data(), p.size);
        return true;
    }

public:

    virtual void device_forward() = 0;

    virtual void device_backprop() = 0;

    template <typename Serializer>
    void serialize_members(Serializer& s) const
    {
        serialize_weights_props(s, get_weights());
        serialize_weights_props(s, get_bias());
    }

    template <typename Serializer>
    bool deserialize_members(Serializer& s)
    {
        init(input_shape);
        if (!deserialize_weights_props(s, get_weights()))
            return false;
        return deserialize_weights_props(s, get_bias());
    }
};