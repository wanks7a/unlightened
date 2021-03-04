#pragma once
#include <shape.h>
#include <device_memory.h>
#include <binary_serialization.h>
#include <device_vector.h>
#include <blob.h>

struct weights_properties
{
    size_t size = 0;
    float* ptr = nullptr;
};

class Layer
{
protected:
    float learing_rate = 0.001f;
    shape output_shape;
    shape input_shape;
    bool in_device_memory = false;
    bool update_on_backprop = true;
    blob_view<float> forward_blob;
    blob_view<float> backward_blob;
public:
    void init_base(const shape& input)
    {
        input_shape = input;
        init(input);
    }
    virtual void init(const shape& input) = 0;
    virtual void forward_pass(Layer* prevLayer) = 0;
    virtual void backprop(Layer* layer) = 0;
    virtual const float* get_output() const = 0;
    virtual const float* derivative_wr_to_input() const = 0;

    const blob_view<float>& get_output_as_blob()
    {
        return forward_blob;
    }

    const blob_view<float>& derivative_as_blob()
    {
        return backward_blob;
    }

    virtual weights_properties get_weights() const
    { 
        return weights_properties();
    };

    virtual weights_properties get_weights_deriv() const
    { 
        return weights_properties();
    };

    virtual weights_properties get_bias() const
    {
        return weights_properties();
    };

    virtual weights_properties get_bias_deriv() const
    {
        return weights_properties();
    };

    virtual void serialize(binary_serialization& s) const 
    {
        s << learing_rate << output_shape << input_shape << in_device_memory << update_on_backprop;
    };

    virtual bool deserialize(binary_serialization& s)
    {
        s >> learing_rate >> output_shape >> input_shape >> in_device_memory >> update_on_backprop;
        return true;
    };

    virtual void pre_epoch(size_t epoch)
    {
    }

    virtual void post_epoch(size_t epoch)
    {
    }

    virtual ~Layer() = default;

    void set_learning_rate(float rate)
    {
        if (rate > 0)
        {
            learing_rate = rate;
        }
    }

    float get_learning_rate() const
    {
        return learing_rate;
    }

    void set_update_weights(bool flag)
    {
        update_on_backprop = flag;
    }

    bool is_frozen() const
    {
        return update_on_backprop;
    }

    cuVector<float> get_device_output()
    {
        cuVector<float> result;
        if(!in_device_memory)
            result.setValues(get_output(), output_shape.size());
        return result;
    }

    void get_device_output(cuVector<float>& v)
    {
        if (!in_device_memory)
            v.setValues(get_output(), output_shape.size());
    }

    std::vector<float> get_native_output()
    {
        if (in_device_memory)
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
        if (!in_device_memory)
            result.setValues(derivative_wr_to_input(), input_shape.size());
        return result;
    }

    std::vector<float> get_native_derivative()
    {
        std::vector<float> result;
        if (in_device_memory)
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
        return in_device_memory;
    }

    shape get_shape() const
    {
        return output_shape;
    }

    size_t output_size() const { return output_shape.size(); }
};

