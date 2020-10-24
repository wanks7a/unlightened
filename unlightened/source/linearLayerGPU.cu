#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <LinearLayerGPU.h>
#include <GpuUtils.h>
#include <algorithm>

#define trPerBlock 1024

template <typename T>
__global__ void k_dense_forward(T* output, const T* weights, const T* input, size_t inputSize, const T* bias, size_t outputSize)
{
    const unsigned int batch_offset_output =  blockIdx.y * outputSize;
    const unsigned int batch_offset_input = blockIdx.y * inputSize;
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < outputSize)
    {
        float result = 0.0f;
        for (int j = 0; j < inputSize; j++)
        {
            //result = __fmaf_rn(input[j], weights[i * inputSize + j], result); // very fast multiply add = a*b + c
            result += input[batch_offset_input + j] * weights[i * inputSize + j];
        }
        output[batch_offset_output + i] = result + bias[i];
    }
}

void linearLayerForwardPassGPU(float* output,const float* weights, const float* input, const shape& input_shape, const float* bias, const shape& output_shape)
{
    auto threadsPerBlock = static_cast<unsigned int>(std::min(output_shape.width, static_cast<size_t>(trPerBlock)));
    auto num_of_blocks = utils::getBlockSize(threadsPerBlock, output_shape.width);
    dim3 blocks(num_of_blocks, output_shape.batches);
    k_dense_forward<float> << <blocks, threadsPerBlock >> > (output, weights, input, input_shape.width, bias, output_shape.width);
    utils::waitAndCheckForErrors();
}

template <typename T>
__global__ void k_calcDerivativeWRtoInput(T* derivativeWRtoInput, size_t inputSize, const T* derivateWRtoOutput, size_t outputSize, const T* weights)
{
    auto inputIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (inputIndex < inputSize)
    {
        derivativeWRtoInput[inputIndex] = 0.0f;
        for (size_t i = 0; i < outputSize; i++)
        {
            derivativeWRtoInput[inputIndex] += derivateWRtoOutput[i] * weights[i * inputSize + inputIndex];
        }
    }
}

void calcDerivativeWRtoInput(float* derivativeWRtoInput, size_t inputSize, const float* derivateWRtoOutput, shape output_shape, const float* weights)
{
    auto threadsPerBlock = static_cast<unsigned int>(std::min(inputSize, static_cast<size_t>(trPerBlock)));
    auto blocks = utils::getBlockSize(threadsPerBlock, inputSize);
    std::vector<cudaStream_t> streams;
    size_t outputSize = output_shape.volume();

    streams.resize(output_shape.batches);
    for (size_t i = 0; i <  streams.size(); i++)
    {
        cudaStreamCreate(&streams[i]);
        k_calcDerivativeWRtoInput << <blocks, threadsPerBlock,0, streams[i]>> > (derivativeWRtoInput + i * inputSize, inputSize, derivateWRtoOutput + i * output_shape.volume(), outputSize, weights);
    }
    utils::waitAndCheckForErrors();
    for (size_t i = 0; i < streams.size(); i++)
    {
        cudaStreamDestroy(streams[i]);
    }
}

template <typename T>
__global__ void k_updateWeights(T* weights, const T* derivativeWRtoOutput,const T* input, size_t inputSize, size_t outputSize, float learning_rate, size_t batches, size_t out_offset)
{
    size_t weightIndex = blockIdx.x * blockDim.x + threadIdx.x;
    size_t derivOutIdx = weightIndex / inputSize;
    size_t inpIdx = weightIndex - derivOutIdx * inputSize;
    if(weightIndex < (inputSize * outputSize))
    {
        float error = 0.0f;
        for (int i = 0; i < batches; i++)
        {
            error += input[inpIdx + i * inputSize] * derivativeWRtoOutput[derivOutIdx + i * out_offset];
        }
        weights[weightIndex] = weights[weightIndex] - learning_rate * error / batches;
    }
}

void updateWeights(float* weights, const float* derivativeWRtoOutput, const float* input, size_t inputSize, size_t outputSize, shape output_shape, float learning_rate)
{
    auto threadsPerBlock = static_cast<unsigned int>(std::min(outputSize * inputSize, static_cast<size_t>(trPerBlock)));
    auto blocks = utils::getBlockSize(threadsPerBlock, outputSize * inputSize);

    k_updateWeights << <blocks, threadsPerBlock >> > (weights, derivativeWRtoOutput, input, inputSize, outputSize, learning_rate, output_shape.batches, output_shape.volume());
    
    utils::waitAndCheckForErrors();
}

template <typename T>
__global__ void k_updateBias(T* bias, const T* derivative_wr_to_out, size_t output_size, float learning_rate, size_t batches, size_t out_offset)
{
    size_t biasIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (biasIndex < output_size)
    {
        float error = 0.0f;
        for (int i = 0; i < batches; i++)
        {
            error += derivative_wr_to_out[biasIndex + i * out_offset];
        }
        bias[biasIndex] = bias[biasIndex] - learning_rate * error / batches;
    }
}

void updateBias(float* bias, const float* derivative_wr_to_out, size_t output_size, shape output_shape, float learning_rate)
{
    auto threadsPerBlock = static_cast<unsigned int>(std::min(output_size, static_cast<size_t>(trPerBlock)));
    auto blocks = utils::getBlockSize(threadsPerBlock, output_size);

    k_updateBias << <blocks, threadsPerBlock >> > (bias, derivative_wr_to_out, output_size, learning_rate, output_shape.batches, output_shape.volume());

    utils::waitAndCheckForErrors();
}