#include <activation_layer.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <GpuUtils.h>
#include <math.h>

struct identity
{
    __host__ __device__
    inline float operator()(const float& input) const
    {
        return input;
    }

    __host__ __device__
    inline float operator()(const float& chain_rule_input, const float& activation_output) const
    {
        return chain_rule_input;
    }
};

struct sigmoid
{
    __host__ __device__
    inline float operator()(const float& input) const
    {
        return 1.0f / (1.0f + expf((-1.0f) * input));
    }

    __host__ __device__
    inline float operator()(const float& chain_rule_input, const float& activation_output) const
    {
        return activation_output * (1 - activation_output) * chain_rule_input;
    }
};

struct relu
{
    __host__ __device__
    inline float operator()(const float& input) const
    {
        if (input > 0)
            return  input;
        else
            return 0;
    }

    __host__ __device__
    inline float operator()(const float& chain_rule_input, const float& activation_output) const
    {
        if (activation_output > 0)
            return chain_rule_input;
        else
            return 0.01f;
    }
};


template <typename Func> __global__
void activation(const float* input, shape* input_shape, float* output)
{
    constexpr Func func;
    const unsigned int tr_index = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int batch_offset = input_shape->volume();
    if (tr_index >= batch_offset)
        return;
    const unsigned int batches = input_shape->batches;
    for (unsigned int i = 0; i < batches; i++)
    {
        output[batch_offset * i + tr_index] = func(input[batch_offset * i + tr_index]);
    }
}

template <typename Func> __global__
void activation_derivative(const float* input, shape* input_shape, const float* activation_output ,float* output)
{
    constexpr Func func;
    const unsigned int tr_index = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int batch_offset = input_shape->volume();
    if (tr_index >= batch_offset)
        return;
    const unsigned int batches = input_shape->batches;
    for (unsigned int i = 0; i < batches; i++)
    {
        output[batch_offset * i + tr_index] = func(input[batch_offset * i + tr_index], activation_output[batch_offset * i + tr_index]);
    }
}

void activation_layer::forward_pass(Layer* prevLayer)
{
    const float* input = nullptr;
    cuVector<float> temp;
    if (prevLayer->is_device_layer())
        input = prevLayer->get_output();
    else
    {
        temp.setValues(prevLayer->get_output(), prevLayer->get_shape().size());
        input = temp.get();
    }
    unsigned int blocks = (output_shape.volume() + 255) / 256;
    utils::device_struct<shape> input_shape = output_shape;
    switch (activ_func)
    {
    case activation_function::Sigmoid: activation<sigmoid><<<blocks, 256>>>(input, input_shape.get(), output.get()); break;
    case activation_function::Identity: activation<identity><<<blocks, 256>>>(input, input_shape.get(), output.get()); break;
    case activation_function::ReLU: activation<relu> << <blocks, 256 >> > (input, input_shape.get(), output.get()); break;
    default:
        break;
    }
    utils::waitAndCheckForErrors();
}

void activation_layer::backprop(Layer* layer)
{
    const float* input = nullptr;
    cuVector<float> temp;
    if (layer->is_device_layer())
        input = layer->derivative_wr_to_input();
    else
    {
        temp.setValues(layer->derivative_wr_to_input(), output_shape.size());
        input = temp.get();
    }
    unsigned int blocks = (output_shape.volume() + 255) / 256;
    utils::device_struct<shape> input_shape = output_shape;
    switch (activ_func)
    {
    case activation_function::Sigmoid: activation_derivative<sigmoid> << <blocks, 256 >> > (input, input_shape.get(), output.get(), derivative.get()); break;
    case activation_function::Identity: activation_derivative<identity> << <blocks, 256 >> > (input, input_shape.get(), output.get(), derivative.get()); break;
    case activation_function::ReLU: activation_derivative<relu> << <blocks, 256 >> > (input, input_shape.get(), output.get(), derivative.get()); break;
    default:
        break;
    }
    utils::waitAndCheckForErrors();
}
