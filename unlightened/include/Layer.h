#pragma once
#include <shape.h>
#include <device_memory.h>

class Layer
{
protected:
    float learing_rate = 0.1f;
    shape output_shape;
    bool device_layer;
public:
    virtual void init(const shape& input) = 0;
    virtual void forwardPass(Layer* prevLayer) = 0;
    virtual void backprop(Layer* layer) = 0;
    virtual const float* getOutput() = 0;
    virtual const float* derivativeWithRespectToInput() = 0;
    virtual void printLayer() = 0;
    virtual ~Layer() = default;
    void set_learning_rate(float rate)
    {
        if (rate > 0)
        {
            learing_rate = rate;
        }
    }

    cuVector<float> get_device_output()
    {
        cuVector<float> result;
        if(!device_layer)
            result.setValues(getOutput(), output_shape.size());
        return result;
    }

    std::vector<float> get_native_output()
    {
        std::vector<float> result;
        if (device_layer)
            result = cuVector<float>::from_device_host(getOutput(), output_shape.size());
        return result;
    }

    bool is_device_layer() const
    {
        return device_layer;
    }

    shape get_shape() const
    {
        return output_shape;
    }

    size_t output_size() const { return output_shape.size(); }
};