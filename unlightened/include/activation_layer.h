#include <device_memory.h>
#include <softmax.h>
#include <serializable_interface.h>

class activation_layer : public serializable_layer<activation_layer>
{ 
public:
    enum class activation_function
    {
        ReLU,
        Sigmoid,
        Softmax,
        LeakyReLU,
        tanh,
        Identity
    };
private:
    activation_function activ_func;
    cuVector<float> output;
    cuVector<float> derivative;
    softmax_activation softmax;
    void softmax_output(const float* input, unsigned int th_per_block, unsigned int blocks, shape* output_shape);
    void softmax_derivative(const float* input, shape* input_shape, unsigned int threads_per_block, unsigned int blocks);
public:
    activation_layer();
    activation_layer(activation_function function);
    void init(const shape& input) override;
    void forward_pass(Layer* prevLayer) override;
    void backprop(Layer* layer) override;
    const float* get_output() const override;
    const float* derivative_wr_to_input() const override;

    template <typename Serializer>
    void serialize_members(Serializer& s) const
    {
        s << activ_func;
    }

    template <typename Serializer>
    void deserialize_members(Serializer& s)
    {
        s >> activ_func;
        init(input_shape);
    }

};