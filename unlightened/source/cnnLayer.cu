#include <conv_filter.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <GpuUtils.h>

#define trPerBlock 256

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
                conv_result += input[chunkStartIndex + i * width + j] * filter[d*filter_size*filter_size + i * filter_size + j];
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

__global__
void k_full_conv_2d(const float* input, shape* input_shape, float* output, shape* output_shape, const float* weights, unsigned int filter_size)
{
    const unsigned int tr_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (tr_index >= (output_shape->width * output_shape->height * output_shape->depth))
        return;
    int row = tr_index / output_shape->width;
    int col = tr_index % output_shape->width;
    const int d = tr_index / (output_shape->width * output_shape->height);
    const int offset = filter_size - 1;
    const int width = input_shape->width;
    const int height = input_shape->height;

    float conv_result = 0.0f;
    row = row - d * output_shape->height;
    for (int i = 0; i < filter_size; i++)
    {
        for (int j = 0; j < filter_size; j++)
        {
            int row_input_idx = row - offset + i;
            int col_input_dix = col - offset + j;
            if (row_input_idx < 0 || row_input_idx >= height || col_input_dix < 0 || col_input_dix >= width)
            {
            }
            else
            {
                conv_result += input[d * width * height + width * row_input_idx + col_input_dix] * weights[i * filter_size + j];
            }

        }
    }
    
    output[tr_index] = conv_result;
}

void full_conv_2d(const float* input,const shape& input_shape, float* output, const shape& output_shape, const float* weights, unsigned int filter_size)
{
    unsigned int num_blocks = ((output_shape.size() + trPerBlock - 1) / trPerBlock);
    dim3 blocks(num_blocks);
    utils::device_struct<shape> device_input_shape(input_shape);
    utils::device_struct<shape> device_output_shape(output_shape);
    k_full_conv_2d << <blocks, trPerBlock >> >(input, device_input_shape.get(), output, device_output_shape.get(), weights, filter_size);
    utils::waitAndCheckForErrors();
}