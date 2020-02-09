#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <LinearLayerGPU.h>
#include <GpuUtils.h>
#include <algorithm>

#define trPerBlock 256

template <typename T>
__global__ void k_linearLayerForwardPass(T* output, T* weights, const T* input, size_t inputSize, size_t outputSize)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < outputSize)
    {
        float result = 0.0f;
        for (int j = 0; j < inputSize; j++)
        {
            __fmaf_rn(input[j], weights[i * inputSize + j], result); // very fast multiply add = a*b + c
            //result += input[j] * weights[i * inputSize + j];
        }
        output[i] = result;
    }
}

void linearLayerForwardPassGPU(float* output, float* weights, const float* input, size_t inputSize, size_t outputSize)
{
    auto threadsPerBlock = static_cast<unsigned int>(std::min(outputSize, static_cast<size_t>(trPerBlock)));
    auto blocks = utils::getBlockSize(threadsPerBlock, outputSize);
    k_linearLayerForwardPass << <blocks, threadsPerBlock >> > (output, weights, input, inputSize, outputSize);
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

void calcDerivativeWRtoInput(float* derivativeWRtoInput, size_t inputSize, const float* derivateWRtoOutput, size_t outputSize, const float* weights)
{
    auto threadsPerBlock = static_cast<unsigned int>(std::min(inputSize, static_cast<size_t>(trPerBlock)));
    auto blocks = utils::getBlockSize(threadsPerBlock, inputSize);
    k_calcDerivativeWRtoInput << <blocks, threadsPerBlock >> > (derivativeWRtoInput, inputSize, derivateWRtoOutput, outputSize, weights);
    utils::waitAndCheckForErrors();
}

template <typename T>
__global__ void k_updateWeightsAndBias(T* weights, const T* derivativeWRtoOutput,const T* input, size_t inputSize, size_t outputSize)
{
    float learning_rate = 0.1f;
    size_t neuronIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if( neuronIndex < outputSize )
    {
        for (size_t i = 0; i < inputSize; i++)
        {
            weights[neuronIndex * inputSize + i] = weights[neuronIndex * inputSize + i] - learning_rate * input[i] * derivativeWRtoOutput[neuronIndex];
        }
    }
}

void updateWeightsAndBias(float* weights, const float* derivativeWRtoOutput, const float* input, size_t inputSize, size_t outputSize)
{
    auto threadsPerBlock = static_cast<unsigned int>(std::min(outputSize, static_cast<size_t>(trPerBlock)));
    auto blocks = utils::getBlockSize(threadsPerBlock, outputSize);
    k_updateWeightsAndBias << <blocks, threadsPerBlock >> > (weights, derivativeWRtoOutput, input, inputSize, outputSize);
    utils::waitAndCheckForErrors();
}