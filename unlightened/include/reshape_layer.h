#include <Layer.h>

class reshape_layer : public Layer
{
    const float* output;
    const float* derivative;
public:
    reshape_layer(shape out_shape);
    void init(const shape& input) override;
    void forward_pass(Layer* prevLayer) override;
    void backprop(Layer* layer) override;
    const float* get_output() override;
    const float* derivative_wr_to_input() override;
};