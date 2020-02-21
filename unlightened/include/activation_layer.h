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
public:
    activation_layer(activation_function function);

    void init(const shape& input) override;
    void forwardPass(Layer* prevLayer) override;
    void backprop(Layer* layer) override;
    const float* getOutput() override;
    const float* derivativeWithRespectToInput() override;
    void printLayer() override;
};