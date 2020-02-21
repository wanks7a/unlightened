#include <activation_layer.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <GpuUtils.h>
#include <math.h>

struct identity
{
    __host__ __device__
    float operator()(const float& input) const
    {
        return input;
    }
};

struct relu
{

};

struct sigmoid
{
    __host__ __device__
    float operator()(const float& input) const
    {
        return 1.0f / (1.0f + expf((-1.0f) * input));
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

void activation_layer::forwardPass(Layer* prevLayer)
{
    const float* input = nullptr;
    cuVector<float> temp;
    if (prevLayer->is_device_layer())
        input = prevLayer->getOutput();
    else
    {
        temp.setValues(prevLayer->getOutput(), prevLayer->get_shape().size());
        input = temp.get();
    }
    unsigned int blocks = (output_shape.volume() + 255) / 256;
    utils::device_struct<shape> input_shape = output_shape;
    switch (activ_func)
    {
    case activation_function::Sigmoid: activation<sigmoid><<<blocks, 256>>>(input, input_shape.get(), output.get()); break;
    case activation_function::Identity: activation<identity><<<blocks, 256>>>(input, input_shape.get(), output.get()); break;
    default:
        break;
    }
    utils::waitAndCheckForErrors();
}

void activation_layer::backprop(Layer* layer)
{

}
