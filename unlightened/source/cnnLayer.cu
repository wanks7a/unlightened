#include <conv_filter.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <GpuUtils.h>

#define trPerBlock 256

template <bool SAME> __global__
void k_conv_3d(const float* input, shape* input_shape, float* output, shape* output_shape, const float* weights, unsigned int filter_size)
{
    const unsigned int tr_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (tr_index >= (output_shape->width * output_shape->height * output_shape->depth))
        return;
    int row = tr_index / output_shape->width;
    int col = tr_index % output_shape->width;
    const int d = tr_index / (output_shape->width * output_shape->height);
    const unsigned int batch_offset_input = blockIdx.y * input_shape->volume();
    const unsigned int batch_offset_output = blockIdx.y * output_shape->volume();
    const int offset = SAME ? (filter_size - 1) / 2 : 0;
    const int width = input_shape->width;
    const int height = input_shape->height;
    const unsigned int filter_offset = d * filter_size * filter_size;

    float conv_result = 0.0f;
    row = row - d * output_shape->height;
    for (int i = 0; i < filter_size; i++)
    {
        for (int j = 0; j < filter_size; j++)
        {
            int row_input_idx = row - offset + i;
            int col_input_dix = col - offset + j;
            if (SAME && (row_input_idx < 0 || row_input_idx >= height || col_input_dix < 0 || col_input_dix >= width))
            {
            }
            else
            {
                conv_result += input[batch_offset_input + d * width * height + width * row_input_idx + col_input_dix] * weights[filter_offset + i * filter_size + j];
            }

        }
    }

    output[batch_offset_output + tr_index] = conv_result;
}

void conv_3d(const float* input, const shape& input_shape, float* output, const shape& output_shape, const float* weights, unsigned int filter_size, bool same)
{
    unsigned int num_blocks = ((output_shape.volume() + trPerBlock - 1) / trPerBlock);
    dim3 blocks(num_blocks, input_shape.batches);
    utils::device_struct<shape> device_input_shape(input_shape);
    utils::device_struct<shape> device_output_shape(output_shape);
    if(same)
        k_conv_3d<true><< <blocks, trPerBlock >> > (input, device_input_shape.get(), output, device_output_shape.get(), weights, filter_size);
    else
        k_conv_3d<false> << <blocks, trPerBlock >> > (input, device_input_shape.get(), output, device_output_shape.get(), weights, filter_size);
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
    const unsigned int batch_offset_input = blockIdx.y * input_shape->volume();
    const unsigned int batch_offset_output = blockIdx.y * output_shape->volume();
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
                conv_result += input[batch_offset_input + d * width * height + width * row_input_idx + col_input_dix] * weights[i * filter_size + j];
            }

        }
    }
    
    output[batch_offset_output + tr_index] = conv_result;
}

void full_conv_2d(const float* input,const shape& input_shape, float* output, const shape& output_shape, const float* weights, unsigned int filter_size)
{
    unsigned int num_blocks = ((output_shape.volume() + trPerBlock - 1) / trPerBlock);
    dim3 blocks(num_blocks, input_shape.batches);
    utils::device_struct<shape> device_input_shape(input_shape);
    utils::device_struct<shape> device_output_shape(output_shape);
    k_full_conv_2d << <blocks, trPerBlock >> >(input, device_input_shape.get(), output, device_output_shape.get(), weights, filter_size);
    utils::waitAndCheckForErrors();
}