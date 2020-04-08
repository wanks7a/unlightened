#include <shape.h>
#include <Layer.h>

void max_pool(const float* input, const shape& input_shape, float* output, const shape& output_shape, char* mask, int filter_size);

void max_pool_backprop(const float* input, const shape& input_shape, float* output, const shape& output_shape, char* mask, int filter_size);

class max_pool : public Layer
{
    virtual void init(const shape& input) override;
    virtual void forward_pass(Layer* prevLayer) override;
    virtual void backprop(Layer* layer) override;
    virtual const float* get_output() override;
    virtual const float* derivative_wr_to_input() override;
    ~max_pool() = default;
};