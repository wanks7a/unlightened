#pragma once
#include <gtest/gtest.h>
#include <Layer.h>

struct test_layer : public Layer
{
    test_layer()
    {
        device_layer = true;
    }

    weights_properties w_props;
    weights_properties d_props;
    weights_properties bias_props;
    weights_properties bias_deriv_props;

    cuVector<float> output;
    void init(const shape& input) override {};
    void forward_pass(Layer* prevLayer) override {};
    void backprop(Layer* layer) override {};
    const float* get_output() const override
    {
        return output.get();
    };
    const float* derivative_wr_to_input() const override
    {
        return output.get();
    };

    void set_output_shape(const shape& sh)
    {
        output_shape = sh;
        input_shape = sh;
    }

    weights_properties get_weights() const override
    {
        return w_props;
    };

    weights_properties get_weights_deriv() const override
    {
        return d_props;
    };

    weights_properties get_bias() const override
    {
        return bias_props;
    };

    weights_properties get_bias_deriv() const override
    {
        return bias_deriv_props;
    };

};

template <typename T>
void compare_vectors(const std::vector<T>& v1, const std::vector<T>& v2)
{
    EXPECT_EQ(v1.size(), v2.size());
    for (size_t i = 0; i < v1.size(); i++)
    {
        EXPECT_EQ(v1[i], v2[i]);
    }
}
