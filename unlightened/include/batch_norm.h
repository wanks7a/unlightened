#pragma once
#include <device_layer.h>
#include <serializable_interface.h>

template <typename Device>
class batch_norm : public  device_layer<batch_norm<Device>, Device, float, true>
{
public:

    virtual void device_forward() = 0;

    virtual void device_backprop() = 0;

    template <typename Serializer>
    void serialize_members(Serializer& s) const
    {

    }

    template <typename Serializer>
    bool deserialize_members(Serializer& s)
    {
        return true;
    }
};