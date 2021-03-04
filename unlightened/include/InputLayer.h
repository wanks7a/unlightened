#pragma once
#include <vector>
#include <iostream>
#include <serializable_interface.h>
#include <device_vector.h>
#include <cuda_device.h>

class InputLayer : public serializable_layer<InputLayer>
{
    std::vector<float> output;
    device_vector<cuda_device, float> out_device;
public:
    InputLayer() { in_device_memory = true; };
    InputLayer(shape shape)
    {
        in_device_memory = true;
        input_shape = shape;
        output_shape = shape;
        output.resize(output_shape.size());
        out_device.reserve(output_shape.size());
    }

    void init(const shape& input) override
    {
    }

    bool set_input(const float* data, size_t size)
    {
        if (size != input_shape.size())
        {
            std::cout << "The data size is not the same as the input shape size." << std::endl;
            return false;
        }
       // memcpy(output.data(), data, size * sizeof(float));
        out_device.memcpy(data, size);
        output = out_device.to_vector();
        return true;
    }

    bool set_input(const std::vector<float>& data)
    {
        if (data.size() != input_shape.size())
        {
            std::cout << "The data size is not the same as the input shape size." << std::endl;
            return false;
        }
        output = data;
        out_device.set_data(output);
        return true;
    }

    void forward_pass(Layer* prevLayer) override
    {
    }

    void backprop(Layer* layer) override
    {
    }

    const float* derivative_wr_to_input() const override
    {
        return nullptr;
    }

    const float* get_output() const override
    {
        return out_device.data();
    };

    template <typename Serializer>
    void serialize_members(Serializer& s) const
    {
    }

    template <typename Serializer>
    void deserialize_members(Serializer& s)
    {
        output.resize(output_shape.size());
    }
};