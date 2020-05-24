#include <activation_layer.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <GpuUtils.h>
#include <math.h>
#include <generic_functions.h>

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
        return activation_output * (1.0f - activation_output) * chain_rule_input;
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
            return 0.0f;
    }
};

struct leaky_relu
{
    __host__ __device__
        inline float operator()(const float& input) const
    {
        if (input >= 0)
            return  input;
        else
            return input * 0.0001f;
    }

    __host__ __device__
        inline float operator()(const float& chain_rule_input, const float& activation_output) const
    {
        if (activation_output >= 0)
            return chain_rule_input;
        else
            return 0.0001f;
    }
};


struct exponent
{
    __host__ __device__
    inline float operator()(const float& input) const
    {
        return expf(input);
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
void activation_derivative(const float* input, shape* input_shape, const float* activation_output, float* output)
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
    unsigned int threads_per_block = 256;
    unsigned int blocks = (output_shape.volume() + 255) / threads_per_block;
    utils::device_struct<shape> input_shape = output_shape;
    switch (activ_func)
    {
    case activation_function::Sigmoid: activation<sigmoid><<<blocks, threads_per_block >>>(input, input_shape.get(), output.get()); break;
    case activation_function::Identity: activation<identity><<<blocks, threads_per_block >>>(input, input_shape.get(), output.get()); break;
    case activation_function::ReLU: activation<relu> << <blocks, threads_per_block >> > (input, input_shape.get(), output.get()); break;
    case activation_function::Softmax: softmax_output(input, threads_per_block, blocks, input_shape.get()); break;
    case activation_function::LeakyReLU: activation<leaky_relu> << <blocks, threads_per_block >> > (input, input_shape.get(), output.get()); break;
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
    unsigned int threads_per_block = 256;
    unsigned int blocks = (output_shape.volume() + 255) / threads_per_block;
    utils::device_struct<shape> input_shape = output_shape;
    switch (activ_func)
    {
    case activation_function::Sigmoid: activation_derivative<sigmoid> << <blocks, threads_per_block >> > (input, input_shape.get(), output.get(), derivative.get()); break;
    case activation_function::Identity: activation_derivative<identity> << <blocks, threads_per_block >> > (input, input_shape.get(), output.get(), derivative.get()); break;
    case activation_function::ReLU: activation_derivative<relu> << <blocks, threads_per_block >> > (input, input_shape.get(), output.get(), derivative.get()); break;
    case activation_function::Softmax: softmax_derivative(input, input_shape.get(), threads_per_block, blocks); break;
    case activation_function::LeakyReLU: activation_derivative<leaky_relu> << <blocks, threads_per_block >> > (input, input_shape.get(), output.get(), derivative.get()); break;
    }
    utils::waitAndCheckForErrors();
}

__global__
void softmax_activation(const float* input, shape* input_shape, float* output, const float* sum_exponents)
{
    const unsigned int tr_index = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int batch_offset = input_shape->volume();
    if (tr_index >= batch_offset)
        return;
    const unsigned int batches = input_shape->batches;
    for (unsigned int i = 0; i < batches; i++)
    {
        output[batch_offset * i + tr_index] = input[batch_offset * i + tr_index] / sum_exponents[i];
    }
}

void activation_layer::softmax_output(const float* input, unsigned int th_per_block, unsigned int blocks, shape* output_shape)
{
    activation<exponent> <<<blocks, th_per_block>>> (input, output_shape, softmax.exponents.get());
    softmax.exponents_sum.memset(0);
    sum_all_values(input_shape, softmax.exponents.get(), softmax.exponents_sum.get());
    softmax_activation <<<blocks, th_per_block >>>(softmax.exponents.get(), output_shape, output.get(), softmax.exponents_sum.get());
}

__global__
void softmax_derivative_calc(const float* input, shape* input_shape, const float* exponents,const float* exponents_sum, float* output)
{
    const unsigned int tr_index = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int batch_offset = input_shape->volume();
    if (tr_index >= batch_offset)
        return;
    const unsigned int batches = input_shape->batches;
    for (unsigned int i = 0; i < batches; i++)
    {
        float current_exponent = exponents[batch_offset * i + tr_index];
        float current_exponnent_sum = exponents_sum[i];
        current_exponent = (current_exponent * (current_exponnent_sum - current_exponent)) / (current_exponnent_sum * current_exponnent_sum);
        output[batch_offset * i + tr_index] = input[batch_offset * i + tr_index] * current_exponent;
    }
}


void activation_layer::softmax_derivative(const float* input, shape* out_shape, unsigned int threads_per_block, unsigned int blocks)
{
    softmax_derivative_calc <<< blocks, threads_per_block >>> (input, out_shape, softmax.exponents.get(), softmax.exponents_sum.get(), derivative.get());
}