#pragma once
#include <Layer.h>
#include <conv_filter.h>

class cnn_layer : public Layer
{
    filter_options options;
    std::vector<filter_conv2d> filters;
public:
    cnn_layer(size_t filter_dimension, size_t num_of_filters);
    void init(const shape& input) override;
    void forwardPass(Layer* prevLayer) override;
    void backprop(Layer* layer) override;
    const float* getOutput() override;
    const float* derivativeWithRespectToInput() override;
    void printLayer() override;
};