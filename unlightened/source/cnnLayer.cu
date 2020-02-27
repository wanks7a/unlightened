#include <conv_filter.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <GpuUtils.h>

#define trPerBlock 256

__global__
void full_convoltion_3d_filter(const float* input,const shape* input_shape,
    float* output, const shape* output_shape,
    const float* weights,
    const float* bias,
    unsigned int filter_row, unsigned int filter_col, unsigned int offset_filter)
{
    const unsigned int trIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int width = output_shape->width;
    const unsigned int height = output_shape->height;
    if (trIndex >= (width * height))
        return;
    const unsigned int input_width = input_shape->width;
    const unsigned int input_height = input_shape->height;
    const unsigned int input_depth = input_shape->depth;
    const unsigned int input_depth_offset = input_shape->area();
    const unsigned int input_batch_offset = input_shape->volume() * blockIdx.y;

    unsigned int row = trIndex / width;
    unsigned int col = trIndex - row * width;
    const int offset = filter_row - offset_filter;

    float conv_result = 0.0f;
    for (int depth = 0; depth < input_depth; depth++)
    {
        for (int i = 0; i < filter_row; i++)
        {
            for (int j = 0; j < filter_col; j++)
            {
                int row_input_idx = row - offset + i;
                int col_input_dix = col - offset + j;
                if (row_input_idx < 0 || row_input_idx >= input_height || col_input_dix < 0 || col_input_dix >= input_width)
                {
                }
                else
                {
                    conv_result += input[input_batch_offset + input_depth_offset * depth + input_width * row_input_idx + col_input_dix] * weights[depth * filter_row * filter_col + i * filter_row + j];
                }

            }
        }       
        conv_result += bias[depth];
    }

    output[output_shape->volume() * blockIdx.y + trIndex] = conv_result;
}

void conv_3d(const float* input, const shape& input_shape, float* output, const shape& output_shape, const float* weights, const float* bias, unsigned int filter_row, unsigned int filter_col, unsigned int offset)
{
    unsigned int num_blocks = ((output_shape.volume() + trPerBlock - 1) / trPerBlock);
    dim3 blocks(num_blocks, output_shape.batches);
    utils::device_struct<shape> device_input_shape(input_shape);
    utils::device_struct<shape> device_output_shape(output_shape);
    full_convoltion_3d_filter<<<blocks, trPerBlock>>>(input, device_input_shape.get(), output, device_output_shape.get(), weights, bias, filter_row, filter_col, offset);
    utils::waitAndCheckForErrors();
}

__global__
void backprop_weights(const float* input, shape* input_shape, float* output, shape* output_shape, const float* weights, 
    unsigned int filter_row, 
    unsigned int filter_col, 
    unsigned int filter_offset, 
    unsigned int weights_offset_batch)
{
    const unsigned int tr_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (tr_index >= (output_shape->width * output_shape->height * output_shape->depth))
        return;
    int row = tr_index / output_shape->width;
    int col = tr_index - row * output_shape->width;
    const int d = tr_index / (output_shape->width * output_shape->height);
    const unsigned int batch_offset_input = blockIdx.y * input_shape->volume();
    const unsigned int batch_offset_output = blockIdx.y * output_shape->volume();
    const int offset = filter_row - filter_offset;
    const int width = input_shape->width;
    const int height = input_shape->height;

    float conv_result = 0.0f;
    row = row - d * output_shape->height;
    for (int i = 0; i < filter_row; i++)
    {
        for (int j = 0; j < filter_col; j++)
        {
            int row_input_idx = row - offset + i;
            int col_input_dix = col - offset + j;
            if (row_input_idx < 0 || row_input_idx >= height || col_input_dix < 0 || col_input_dix >= width)
            {
            }
            else
            {
                conv_result += input[batch_offset_input + d * width * height + width * row_input_idx + col_input_dix] * weights[blockIdx.y * weights_offset_batch + i * filter_row + j];
            }

        }
    }

    output[batch_offset_output + tr_index] = conv_result;
}

void backprop_weights_3d(const float* input,const shape& input_shape, float* output, const shape& output_shape, const float* weights,
    unsigned int filter_row, 
    unsigned int filter_col, 
    unsigned int filter_offset, 
    unsigned int weights_offset_batch)
{
    unsigned int num_blocks = ((output_shape.volume() + trPerBlock - 1) / trPerBlock);
    dim3 blocks(num_blocks, output_shape.batches);
    utils::device_struct<shape> device_input_shape(input_shape);
    utils::device_struct<shape> device_output_shape(output_shape);
    backprop_weights << <blocks, trPerBlock >> > (input, device_input_shape.get(), output, device_output_shape.get(), weights, filter_row, filter_col, filter_offset, weights_offset_batch);
    utils::waitAndCheckForErrors();
}


__global__
void merge_conv_with_bias(const float* input, const shape* input_shape, const float* bias_vector, float* output, const unsigned int batch_offset)
{
    const unsigned int tr_index = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int area = input_shape->area();
    if (tr_index >= area)
        return;
    const unsigned int depth = input_shape->depth;
    const unsigned int batch_offset_input = blockIdx.y * input_shape->volume();
    float result = 0.0f;
    for (unsigned int i = 0; i < depth; i++)
    {
        result += (input[batch_offset_input + i * area +  tr_index] + bias_vector[i]);
    }
    
    output[batch_offset * blockIdx.y + tr_index] = result;
}

void merge_conv_with_bias(const float* input, const shape& input_shape, const float* bias_vector, float* output, const unsigned int batch_offset)
{
    unsigned int num_blocks = ((input_shape.area() + trPerBlock - 1) / trPerBlock);
    dim3 blocks(num_blocks, input_shape.batches);
    utils::device_struct<shape> device_input_shape(input_shape);
    merge_conv_with_bias << <blocks, trPerBlock >> > (input, device_input_shape.get(), bias_vector, output, batch_offset);
    utils::waitAndCheckForErrors();
}

template <bool HORIZONTAL>  __global__
void flip_horizontal(float* input)
{
    if ((blockDim.x / 2) < threadIdx.x)
        return;
    if (HORIZONTAL)
    {
        float temp = input[blockDim.x * blockDim.y * blockIdx.x + blockDim.x * threadIdx.y + blockDim.x - 1 - threadIdx.x];
        input[blockDim.x * blockDim.y * blockIdx.x + blockDim.x * threadIdx.y + blockDim.x - 1 - threadIdx.x] = input[blockDim.x * blockDim.y * blockIdx.x + blockDim.x * threadIdx.y + threadIdx.x];
        input[blockDim.x * blockDim.y * blockIdx.x + blockDim.x * threadIdx.y + threadIdx.x] = temp;
    }
    else
    {
        float temp = input[blockDim.x * blockDim.y * blockIdx.x + blockDim.x * (blockDim.y - 1 - threadIdx.y) + threadIdx.x];
        input[blockDim.x * blockDim.y * blockIdx.x + blockDim.x * (blockDim.y - 1 - threadIdx.y) + threadIdx.x] = input[blockDim.x * blockDim.y * blockIdx.x + blockDim.x * threadIdx.y + threadIdx.x];
        input[blockDim.x * blockDim.y * blockIdx.x + blockDim.x * threadIdx.y + threadIdx.x] = temp;
    }
}

void flip_filter(float* input,const shape& filter_shape, bool horizontal)
{
    dim3 blocks(filter_shape.depth);
    dim3 threads(filter_shape.width, filter_shape.height);
    if (horizontal)
        flip_horizontal<true><<<blocks, threads>>>(input);
    else
        flip_horizontal<false><<<blocks, threads>>>(input);
    utils::waitAndCheckForErrors();
}

__global__
void k_update_weights(const float* error, float* weights, const shape* weights_shape, float learning_rate)
{
    const unsigned int tr_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (tr_index >= weights_shape->volume())
        return;
    weights[tr_index] = weights[tr_index] - learning_rate * error[tr_index];
}

void update_weights(const float* error, float* weights,const shape& weights_shape, float learning_rate)
{
    unsigned int num_blocks = ((weights_shape.volume() + trPerBlock - 1) / trPerBlock);
    utils::device_struct<shape> device_input_shape(weights_shape);
    k_update_weights << <num_blocks, trPerBlock >> > (error, weights, device_input_shape.get(), learning_rate);
    utils::waitAndCheckForErrors();
}

__global__
void derivative_input(const float* input, shape* input_shape, float* output, shape* output_shape, const float* weights,
                      unsigned int filter_row,
                      unsigned int filter_col,
                      unsigned int filter_offset,
                      unsigned int weights_offset_batch)
{
    const unsigned int tr_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (tr_index >= (output_shape->width * output_shape->height * output_shape->depth))
        return;
    int row = tr_index / output_shape->width;
    int col = tr_index - row * output_shape->width;
    const int d = tr_index / (output_shape->width * output_shape->height);
    const unsigned int batch_offset_output = blockIdx.y * output_shape->volume();
    const int offset = filter_row - filter_offset;
    const int width = input_shape->width;
    const int height = input_shape->height;
    const unsigned int input_shape_volume = input_shape->volume();

    float whole_result = 0.0f;
    row = row - d * output_shape->height;
    for (size_t channel = 0; channel < output_shape->depth; channel++)
    {
        float conv_result = 0.0f;
        for (int i = 0; i < filter_row; i++)
        {
            for (int j = 0; j < filter_col; j++)
            {
                int row_input_idx = row - offset + i;
                int col_input_dix = col - offset + j;
                if (row_input_idx < 0 || row_input_idx >= height || col_input_dix < 0 || col_input_dix >= width)
                {
                }
                else
                {
                    conv_result += input[input_shape_volume * channel + d * width * height + width * row_input_idx + col_input_dix] * weights[blockIdx.y * weights_offset_batch + channel * filter_row * filter_col + i * filter_row + j];
                }

            }
        }
        whole_result += conv_result;
    }
    output[batch_offset_output + tr_index] = whole_result;
}

void derivative_input_3d(const float* input,const shape& input_shape, float* output,const shape& output_shape, const float* weights,
    unsigned int filter_row,
    unsigned int filter_col,
    unsigned int filter_offset,
    unsigned int weights_offset_batch)
{
    unsigned int num_blocks = ((output_shape.volume() + trPerBlock - 1) / trPerBlock);
    dim3 blocks(num_blocks, output_shape.batches);
    utils::device_struct<shape> device_input_shape(input_shape);
    utils::device_struct<shape> device_output_shape(output_shape);
    derivative_input << <blocks, trPerBlock >> > (input, device_input_shape.get(), output, device_output_shape.get(), weights, filter_row, filter_col, filter_offset, weights_offset_batch);
    utils::waitAndCheckForErrors();
}