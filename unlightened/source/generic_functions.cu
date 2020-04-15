#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <generic_functions.h>
#include <GpuUtils.h>
#include <vector>

template <unsigned int NUM_OF_THREADS> __global__
void k_sum_all(const float* input, float* value, size_t max_size)
{
    __shared__ float sh_memory[NUM_OF_THREADS];
    const unsigned int tr_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tr_index >= max_size)
    {
        sh_memory[threadIdx.x] = 0.0f;
        return;
    }
    else
    {
        sh_memory[threadIdx.x] = input[tr_index];
    }

    __syncthreads();
    #pragma unroll
    for (int i = NUM_OF_THREADS / 2; i > 0; i = i / 2)
    {
        if (threadIdx.x < i)
            sh_memory[threadIdx.x] += sh_memory[threadIdx.x + i];
        __syncthreads();
    }
    
    if(threadIdx.x == 0)
        atomicAdd(value, sh_memory[threadIdx.x]);
}

void sum_all_values(const shape& sh, const float* input, float* value)
{
    int num_of_threads = 1024;
    auto blocks = utils::getBlockSize(num_of_threads, sh.volume());
    std::vector<cudaStream_t> streams;
    streams.resize(sh.batches);
    for (size_t i = 0; i < sh.batches; i++)
    {
        cudaStreamCreate(&streams[i]);
        k_sum_all<1024> << <blocks, num_of_threads, 0, streams[i] >> > (input + i * sh.volume(), value + i, sh.volume());
    }
    utils::waitAndCheckForErrors();
    for (size_t i = 0; i < sh.batches; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
}