#pragma once
#include <Layer.h>
#include <conv_filter.h>

void full_conv_2d(const float* input, const shape& input_shape, float* output, const shape& output_shape, const float* weights, unsigned int filter_size);
void conv_3d(const float* input, const shape& input_shape, float* output, const shape& output_shape, const float* weights, unsigned int filter_size, bool same);

class cnn_layer : public Layer
{
    filter_options options;
    std::vector<filter_conv2d> filters;
    size_t filters_size;
    cuVector<float> output;
    Layer* input_layer;
public:
    cnn_layer(size_t filter_dimension, size_t num_of_filters);
    void init(const shape& input) override;
    void forwardPass(Layer* prevLayer) override;
    void backprop(Layer* layer) override;
    const float* getOutput() override;
    const float* derivativeWithRespectToInput() override;
    void printLayer() override;

    filter_options get_options() const
    {
        return options;
    }

    void set_options(const filter_options& opt)
    {
        options = opt;
    }
};