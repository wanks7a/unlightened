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
    int col = tr_index - row * output_shape->width;
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
    dim3 blocks(num_blocks, output_shape.batches);
    utils::device_struct<shape> device_input_shape(input_shape);
    utils::device_struct<shape> device_output_shape(output_shape);
    if(same)
        k_conv_3d<true><< <blocks, trPerBlock >> > (input, device_input_shape.get(), output, device_output_shape.get(), weights, filter_size);
    else
        k_conv_3d<false> << <blocks, trPerBlock >> > (input, device_input_shape.get(), output, device_output_shape.get(), weights, filter_size);
    utils::waitAndCheckForErrors();
}

template <bool KEEP_RESULT = false> __global__
void k_full_conv_2d(const float* input, shape* input_shape, float* output, shape* output_shape, const float* weights, unsigned int filter_size)
{
    const unsigned int tr_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (tr_index >= (output_shape->width * output_shape->height * output_shape->depth))
        return;
    int row = tr_index / output_shape->width;
    int col = tr_index - row * output_shape->width;
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
    if(KEEP_RESULT)
        output[batch_offset_output + tr_index] += conv_result;
    else
        output[batch_offset_output + tr_index] = conv_result;
}

void full_conv_2d(const float* input,const shape& input_shape, float* output, const shape& output_shape, const float* weights, unsigned int filter_size, bool keep_result)
{
    unsigned int num_blocks = ((output_shape.volume() + trPerBlock - 1) / trPerBlock);
    dim3 blocks(num_blocks, output_shape.batches);
    utils::device_struct<shape> device_input_shape(input_shape);
    utils::device_struct<shape> device_output_shape(output_shape);
    if(keep_result)
        k_full_conv_2d<true><< <blocks, trPerBlock >> >(input, device_input_shape.get(), output, device_output_shape.get(), weights, filter_size);
    else
        k_full_conv_2d<false><< <blocks, trPerBlock >> >(input, device_input_shape.get(), output, device_output_shape.get(), weights, filter_size);
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