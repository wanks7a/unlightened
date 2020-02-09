#include <conv_filter.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <GpuUtils.h>

__global__ void k_filter_forwardPass(const float* input, shape* input_shape, const float* weights, float* output, unsigned int filter_size)
{
    float conv_result = 0.0f;
    unsigned int width = input_shape->width;
    unsigned int chunkStartIndex = blockIdx.y* width + blockIdx.x;
    for (unsigned int i = 0; i < filter_size; i++)
    {
        for (unsigned int j = 0; j < filter_size; j++)
        {
            conv_result += input[chunkStartIndex + i * width + j] * weights[i * filter_size + j];
        }
    }
    output[blockDim.x * blockIdx.y + blockIdx.x] = conv_result;
}

void filter_forwardPass(const float* input, shape input_shape, const float* weights, float* output, shape output_shape, unsigned int filter_size)
{
    dim3 blocks(output_shape.width, output_shape.height);
    utils::device_struct<shape> gpu_shape(input_shape);
    k_filter_forwardPass<<<blocks, blocks.x>> > (input, gpu_shape.get(), weights, output, filter_size);
    utils::waitAndCheckForErrors();
}