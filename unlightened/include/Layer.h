#pragma once
#include <shape.h>
#include <device_memory.h>
#include <binary_serialization.h>

class Layer
{
protected:
    float learing_rate = 0.1f;
    shape output_shape;
    shape input_shape;
    bool device_layer = false;
    bool update_on_backprop = true;
public:
    void init_base(const shape& input)
    {
        input_shape = input;
        init(input);
    }
    virtual void init(const shape& input) = 0;
    virtual void forward_pass(Layer* prevLayer) = 0;
    virtual void backprop(Layer* layer) = 0;
    virtual const float* get_output() = 0;
    virtual const float* derivative_wr_to_input() = 0;
    
    virtual void serialize(binary_serialization& s) const 
    {
        s << learing_rate << output_shape << input_shape << device_layer << update_on_backprop;
    };

    virtual bool deserialize(binary_serialization& s)
    {
        s >> learing_rate >> output_shape >> input_shape >> device_layer >> update_on_backprop;
        return true;
    };

    virtual ~Layer() = default;

    void set_learning_rate(float rate)
    {
        if (rate > 0)
        {
            learing_rate = rate;
        }
    }

    void set_update_weights(bool flag)
    {
        update_on_backprop = flag;
    }

    cuVector<float> get_device_output()
    {
        cuVector<float> result;
        if(!device_layer)
            result.setValues(get_output(), output_shape.size());
        return result;
    }

    void get_device_output(cuVector<float>& v)
    {
        if (!device_layer)
            v.setValues(get_output(), output_shape.size());
    }

    std::vector<float> get_native_output()
    {
        if (device_layer)
            return cuVector<float>::from_device_host(get_output(), output_shape.size());
        else
        {
            std::vector<float> result(output_shape.size());
            memcpy(result.data(), get_output(), output_shape.size() * sizeof(float));
            return result;
        }
    }

    cuVector<float> get_device_derivative()
    {
        cuVector<float> result;
        if (!device_layer)
            result.setValues(derivative_wr_to_input(), input_shape.size());
        return result;
    }

    std::vector<float> get_native_derivative()
    {
        std::vector<float> result;
        if (device_layer)
            result = cuVector<float>::from_device_host(derivative_wr_to_input(), input_shape.size());
        else
        {
            result.reserve(input_shape.size());
            for (size_t i = 0; i < input_shape.size(); i++)
            {
                result.emplace_back(derivative_wr_to_input()[i]);
            }
        }
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