#pragma once
#include <vector>
#include <iostream>
#include <serializable_interface.h>

class InputLayer : public serializable_layer<InputLayer>
{
    std::vector<float> output;
public:
    InputLayer() = default;
    InputLayer(shape shape)
    {
        input_shape = shape;
        output_shape = shape;
        output.resize(output_shape.size());
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
        memcpy(output.data(), data, size * sizeof(float));
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
        return true;
    }

    void forward_pass(Layer* prevLayer) override
    {
    }

    void backprop(Layer* layer) override
    {
    }

    const float* derivative_wr_to_input() override
    {
        return nullptr;
    }

    const float* get_output() override
    {
        return output.data();
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