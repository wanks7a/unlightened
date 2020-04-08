#include <shape.h>
#include <Layer.h>

void max_pooling(const float* input, const shape& input_shape, float* output, const shape& output_shape, char* mask, int filter_size);

void max_pooling_backprop(const float* input, const shape& input_shape, float* output, const shape& output_shape, char* mask, int filter_size);

class max_pool : public Layer
{
    cuVector<char> mask;
    cuVector<float> output;
    cuVector<float> derivative;
    cuVector<float> input;
    cuVector<float> derivative_input;
    int filter_size;
public:
    max_pool(int f_size);

    virtual void init(const shape& input) override;
    virtual void forward_pass(Layer* prevLayer) override;
    virtual void backprop(Layer* layer) override;
    virtual const float* get_output() override;
    virtual const float* derivative_wr_to_input() override;
    ~max_pool() = default;
};