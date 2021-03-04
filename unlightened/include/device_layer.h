#pragma once
#include <serializable_interface.h>

template <typename TLayer, typename Device, typename DType, bool KEEP_FORWARD_INPUT>
class device_layer : public serializable_layer<TLayer>
{
    device_vector<Device, DType> forward_data;
    const DType* forward_ptr = nullptr;
    const DType* backprop_ptr = nullptr;
public:
    void forward_pass(Layer* prevLayer) override final
    {
        if (prevLayer->is_device_layer())
        {
            forward_ptr = prevLayer->get_output();
            static_cast<TLayer*>(this)->device_forward();
        }
        else
        {
            forward_data.set_data(prevLayer->get_output(), prevLayer->get_shape().size());
            forward_ptr = forward_data.data();
            static_cast<TLayer*>(this)->device_forward();
            if (!KEEP_FORWARD_INPUT)
                forward_data.clear();
        }
    }

    virtual void backprop(Layer* layer) override final
    {
        device_vector<Device, DType> backprop_buff;
        if (layer->is_device_layer())
        {
            backprop_ptr = layer->derivative_wr_to_input();
            static_cast<TLayer*>(this)->device_backprop();
        }
        else
        {
            backprop_buff.set_data(layer->derivative_wr_to_input(), layer->get_shape().size());
            backprop_ptr = backprop_buff.data();
            static_cast<TLayer*>(this)->device_backprop();
            backprop_ptr = nullptr;
        }
    }
protected:

    const DType* get_forward_input() const
    {
        return forward_ptr;
    }

    const DType* get_backprop_input() const
    {
        return backprop_ptr;
    }

public:
    template <typename Serializer>
    inline void serialize_members(Serializer& s) const
    {
        const TLayer& ref = static_cast<const TLayer&>(*this);
        ref.serialize_members(s);
    }

    template <typename Serializer>
    inline bool deserialize_members(Serializer& s)
    {
        TLayer& ref = static_cast<TLayer&>(*this);
        return ref.deserialize_members(s);
    }

public:
    device_layer()
    {
        in_device_memory = true;
    }

    friend TLayer;
};

template <typename TLayer, typename Device>
class device_layer_new : public serializable_layer<TLayer>
{
    std::shared_ptr<Device> d;
public:
    void forward_pass(Layer* layer) override
    {
        const auto& b = layer->get_output_as_blob();
        // TODO remove this if when every layer uses this interface
        if (b.data() == nullptr)
        {
            blob_view<float> blob_v;
            blob_v.set_size(layer->get_output(), input_shape.size());
            TLayer& l = static_cast<TLayer&>(*this);
            l.forward(d, blob_v);
        }
        if (b.size() == input_shape.size())
        {
            TLayer& l = static_cast<TLayer&>(*this);
            l.forward(d, b);
        }
    }
    
    void backprop(Layer* layer) override
    {
        const auto& b = layer->derivative_as_blob();
        if (b.data() == nullptr)
        {
            blob_view<float> blob_v;
            blob_v.set_size(layer->derivative_wr_to_input(), output_shape.size());
            TLayer& l = static_cast<TLayer&>(*this);
            l.backward(d, blob_v);
        }
        if (b.size() == output_shape.size())
        {
            TLayer& l = static_cast<TLayer&>(*this);
            l.backward(d, b);
        }
    }
};