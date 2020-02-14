#include <conv_filter.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <GpuUtils.h>

__global__ 
void k_conv2d_kernel(const float* input, shape* input_shape, const float* filter, float* output, unsigned int filter_size)
{
    float conv_result = 0.0f;
    unsigned int width = input_shape->width;
    unsigned int height = input_shape->height;
    unsigned int depth = input_shape->depth;
    for (unsigned int d = 0; d < depth; d++)
    {
        unsigned int chunkStartIndex = d*width*height + blockIdx.y * width + blockIdx.x;
        for (unsigned int i = 0; i < filter_size; i++)
        {
            for (unsigned int j = 0; j < filter_size; j++)
            {
                conv_result += input[chunkStartIndex + i * width + j] * filter[i * filter_size + j];
            }
        }
    }
    output[blockDim.x * blockIdx.y + blockIdx.x] = conv_result;
}

void conv2d_kernel(const float* input, const shape& input_shape, const float* weights, float* output, shape output_shape, unsigned int filter_size)
{
    dim3 blocks(static_cast<unsigned int>(output_shape.width), static_cast<unsigned int>(output_shape.height));
    utils::device_struct<shape> gpu_shape(input_shape);
    k_conv2d_kernel << <blocks, blocks.x >> > (input, gpu_shape.get(), weights, output, filter_size);
    utils::waitAndCheckForErrors();
}