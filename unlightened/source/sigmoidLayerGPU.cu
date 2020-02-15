#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <algorithm>
#include <GpuUtils.h>

template <typename T>
__global__ void k_sigmoidLayer(T* input, T* output, size_t inputSize)
{
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < inputSize; i += blockDim.x * gridDim.x)
	{
		output[i] = 1.0f / (1.0f + expf((-1.0f) * input[i]));
	}
}

void sigmoidLayer(float* input, float* output, size_t inputSize)
{
	auto threadsPerBlock = static_cast<unsigned int>(std::min(inputSize, static_cast<size_t>(256)));
	auto blocks = utils::getBlockSize(threadsPerBlock, inputSize);
	k_sigmoidLayer << <blocks, threadsPerBlock >> > (input, output, inputSize);
}

template <typename T>
__global__ void k_sigmoidLayerDerivative(T* derivativeWRtoInput, const T* output, const T* derivativeWRtoOutput, size_t inputSize)
{
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < inputSize; i += blockDim.x * gridDim.x)
	{
		derivativeWRtoInput[i] = output[i] * (1 - output[i]) * derivativeWRtoOutput[i];
	}
}

void sigmoidLayerDerivative(float* derivativeWRtoInput, const float* output, const float* derivativeWRtoOutput, size_t inputSize)
{
	auto threadsPerBlock = static_cast<unsigned int>(std::min(inputSize, static_cast<size_t>(256)));
	auto blocks = utils::getBlockSize(threadsPerBlock, inputSize);
	k_sigmoidLayerDerivative << <blocks, threadsPerBlock >> > (derivativeWRtoInput, output, derivativeWRtoOutput, inputSize);
}