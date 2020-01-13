#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <LinearLayerGPU.h>
#include <GpuUtils.h>
#include <algorithm>

static void waitAndCheckForErrors()
{
    // Check for any errors launching the kernel
    auto cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }
}

template <typename T>
__global__ void k_linearLayerForwardPass(T* output, T* weights, const T* input, size_t inputSize, size_t outputSize)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outputSize; i += blockDim.x * gridDim.x)
    {
        output[i] = 0.0f;
        for (int j = 0; j < inputSize; j++)
        {
            output[i] += input[j] * weights[i * inputSize + j];
        }
    }
}

void linearLayerForwardPassGPU(float* output, float* weights, const float* input, size_t inputSize, size_t outputSize)
{
    auto threadsPerBlock = std::min(outputSize, static_cast<size_t>(256));
    auto blocks = utils::getBlockSize(threadsPerBlock, outputSize);
    k_linearLayerForwardPass << <blocks, threadsPerBlock >> > (output, weights, input, inputSize, outputSize);
    waitAndCheckForErrors();
}

template <typename T>
__global__ void k_calcDerivativeWRtoInput(T* derivativeWRtoInput, size_t inputSize, const T* derivateWRtoOutput, size_t outputSize, const T* weights)
{
    for (size_t inputIndex = blockIdx.x * blockDim.x + threadIdx.x; inputIndex < inputSize; inputIndex += blockDim.x * gridDim.x)
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
    auto threadsPerBlock = std::min(inputSize, static_cast<size_t>(256));
    auto blocks = utils::getBlockSize(threadsPerBlock, inputSize);
    k_calcDerivativeWRtoInput << <blocks, threadsPerBlock >> > (derivativeWRtoInput, inputSize, derivateWRtoOutput, outputSize, weights);
    waitAndCheckForErrors();
}

template <typename T>
__global__ void k_updateWeightsAndBias(T* weights, const T* derivativeWRtoOutput,const T* input, size_t inputSize, size_t outputSize)
{
    float learning_rate = 0.1f;
    for (size_t neuronIndex = blockIdx.x * blockDim.x + threadIdx.x; neuronIndex < outputSize; neuronIndex += blockDim.x * gridDim.x)
    {
        for (size_t i = 0; i < inputSize; i++)
        {
            weights[neuronIndex * inputSize + i] = weights[neuronIndex * inputSize + i] - learning_rate * input[i] * derivativeWRtoOutput[neuronIndex];
        }
    }
}

void updateWeightsAndBias(float* weights, const float* derivativeWRtoOutput, const float* input, size_t inputSize, size_t outputSize)
{
    auto threadsPerBlock = std::min(outputSize, static_cast<size_t>(256));
    auto blocks = utils::getBlockSize(threadsPerBlock, outputSize);
    k_updateWeightsAndBias << <blocks, threadsPerBlock >> > (weights, derivativeWRtoOutput, input, inputSize, outputSize);
    waitAndCheckForErrors();
}

namespace utils
{
    size_t getBlockSize(size_t threadsPerBlock, size_t maxThreads)
    {
        return static_cast<size_t>(std::ceil(static_cast<double>(maxThreads) / threadsPerBlock));
    }

    bool GpuInit()
    {
        // Choose which GPU to run on, change this on a multi-GPU system.
        auto cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!");
            return false;
        }
        return true;
    }

    bool GpuRelase()
    {
        auto cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return false;
        }
        return true;
    }
}