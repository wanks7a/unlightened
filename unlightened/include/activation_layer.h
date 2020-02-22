#include <Layer.h>
#include <device_memory.h>

class activation_layer : public Layer
{ 
public:
    enum class activation_function
    {
        ReLU,
        Sigmoid,
        Identity
    };
private:
    activation_function activ_func;
    cuVector<float> output;
    cuVector<float> derivative;
public:
    activation_layer(activation_function function);

    void init(const shape& input) override;
    void forward_pass(Layer* prevLayer) override;
    void backprop(Layer* layer) override;
    const float* get_output() override;
    const float* derivative_wr_to_input() override;
    void printLayer() override;
};