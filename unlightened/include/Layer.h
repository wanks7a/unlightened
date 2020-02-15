#pragma once
#include <shape.h>

class Layer
{
protected:
    float learing_rate = 0.1f;
    shape output_shape;
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
        if (rate < 1.0f && rate > 0)
        {
            learing_rate = rate;
        }
    }

    shape get_shape() const
    {
        return output_shape;
    }

    size_t output_size() const { return output_shape.size(); }
};