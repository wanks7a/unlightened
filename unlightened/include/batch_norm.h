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
protected:

    virtual std::vector<std::vector<float>> serialize_additional_members() const = 0;
    virtual bool deserialize_additional_members(const std::vector<std::vector<float>>& values) = 0;

public:

    virtual void device_forward() = 0;

    virtual void device_backprop() = 0;

    template <typename Serializer>
    void serialize_members(Serializer& s) const
    {
        serialize_weights_props(s, get_weights());
        serialize_weights_props(s, get_bias());
        auto additional_members = serialize_additional_members();
        s << additional_members;
    }

    template <typename Serializer>
    bool deserialize_members(Serializer& s)
    {
        init(input_shape);
        if (!deserialize_weights_props(s, get_weights()))
            return false;
        if (!deserialize_weights_props(s, get_bias()))
            return false;
        std::vector<std::vector<float>> additional_members;
        s >> additional_members;
        return deserialize_additional_members(additional_members);
    }
};