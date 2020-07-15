#pragma once
#include <conv2d_cudnn.h>

class conv2d_transposed : protected conv2d_cudnn
{
public:
    conv2d_transposed(const filter_options& opt);
    void init(const shape& input) override;
    void forward_pass(Layer* prevLayer) override;
    void backprop(Layer* layer) override;
    const float* get_output() override;
    const float* derivative_wr_to_input() override;
private:
    void forward_pass();
};