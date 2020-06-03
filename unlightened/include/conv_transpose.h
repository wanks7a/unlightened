#include <cnn_layer.h>

class conv2d_transposed : protected cnn_layer
{
    void forward_pass(Layer* prevLayer) override;
    void backprop(Layer* layer) override;
    const float* get_output() override;
    const float* derivative_wr_to_input() override;
};