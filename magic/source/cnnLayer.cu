#include <conv_filter.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <GpuUtils.h>

template <unsigned int N, unsigned int Padding>
__global__ void k_filter_forwardPass(const float* input, shape* input_shape, size_t padding, float* output)
{
    __shared__ float chunk[N + 2 * Padding][N + 2 * Padding];
    size_t y_pad = (gridDim.x * blockDim.x) % input_shape->width;
    size_t x_pad = (gridDim.y * blockDim.y) % input_shape->height;

    #pragma unroll N
    for (int i = 0; i < N; i++)
    {
        #pragma unroll N
        for (int j = 0; j < N; j++)
        {
            chunk[threadIdx.y + i][threadIdx.x + j] = 0.0f;
        }
    }

    if (y_pad > 0 && blockIdx.x == (gridDim.x - 1) && threadIdx.x > (blockDim.x - y_pad - 1))
    {
        return;
    }
        
    if (x_pad > 0 && blockIdx.y == (gridDim.y - 1) && threadIdx.y > (blockDim.y - x_pad - 1))
    {
        return;
    }
 
    size_t chunkStartIndex = blockDim.x * blockDim.y * gridDim.x * blockIdx.y + blockIdx.x * blockDim.x - y_pad * blockIdx.y * blockDim.y;
    chunk[threadIdx.y + Padding][threadIdx.x + Padding] = input[chunkStartIndex + input_shape->width * threadIdx.y + threadIdx.x];
    __syncthreads();
    output[chunkStartIndex + threadIdx.y * input_shape->width + threadIdx.x] = chunk[threadIdx.y + Padding][threadIdx.x + Padding];
}

void filter_forwardPass(const float* input, shape input_shape, size_t padding, float* output)
{
    dim3 threadsPerBlock(3, 3);
    dim3 blocks(2, 3);
    utils::device_struct<shape> gpu_shape(input_shape);
    k_filter_forwardPass<3,1><<<blocks, threadsPerBlock >> > (input, gpu_shape.get(), padding, output);
    utils::waitAndCheckForErrors();
}