#include <conv_filter.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <GpuUtils.h>

#define trPerBlock 256
#define biasThreadPerBlock 1024

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

template <bool HORIZONTAL_LINES>  __global__
void flip_horizontal(float* input, const shape* input_shape)
{
    const unsigned int tr_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (tr_index >= input_shape->size())
        return;
    unsigned int area = input_shape->area();
    unsigned int width = input_shape->width;
    unsigned int row = tr_index / width;
    unsigned int col = tr_index - row * width;
    const int d = tr_index / area;
    if (!HORIZONTAL_LINES)
    {
        if ((width / 2) < col)
            return;

        float temp = input[width * row + col];
        input[width * row + col] = input[width * row + width - 1 - col];
        input[width * row + width - 1 - col] = temp;
        
    }
    else
    {
        unsigned int height = input_shape->height;
        unsigned int relative_row = row - d * height;
        if (relative_row < (height / 2))
        {
            float temp = input[width * row + col];
            input[width * row + col] = input[width * (row - relative_row + height - 1 - relative_row) + col];
            input[width * (row - relative_row + height - 1 - relative_row) + col] = temp;
        }
    }
}

void flip_filter(float* input,const shape& filter_shape, bool horizontal_lines)
{
    unsigned int num_blocks = ((filter_shape.size() + trPerBlock - 1) / trPerBlock);
    utils::device_struct<shape> device_input_shape(filter_shape);
    if (horizontal_lines)
        flip_horizontal<true><<<num_blocks, trPerBlock>>>(input, device_input_shape.get());
    else
        flip_horizontal<false><<<num_blocks, trPerBlock>>>(input, device_input_shape.get());
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

    float conv_result = 0.0f;
    row = row - d * output_shape->height;
    // here we are using the batches for the number of filters
    for (size_t channel = 0; channel < input_shape->batches; channel++)
    {
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
    }
    output[batch_offset_output + tr_index] = conv_result;
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

__global__
void update_weights_kernel(const float* weights_error, shape* weights_shape, float* weights, float learning_rate)
{
    const unsigned int tr_index = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int batches = weights_shape->batches;
    const unsigned int offset = weights_shape->size();
    const unsigned int per_batch_offset = weights_shape->volume();

    if (tr_index >= per_batch_offset)
        return;

    float error = 0.0f;
    for (int batch = 0; batch < batches; batch++)
    {
        error += weights_error[blockIdx.y * offset + batch * per_batch_offset + tr_index];
    }
    
    error = learning_rate * (error / batches);
    
    weights[blockIdx.y * per_batch_offset + tr_index] -= error;
}

void update_weights(const float* weights_error, shape weights_shape, unsigned int num_of_filters, float* weights, float learning_rate)
{
    unsigned int num_blocks = ((weights_shape.volume() + trPerBlock - 1) / trPerBlock);
    dim3 blocks(num_blocks, num_of_filters);
    utils::device_struct<shape> device_input_shape(weights_shape);

    update_weights_kernel << <blocks, trPerBlock >> > (weights_error, device_input_shape.get(), weights, learning_rate);

    utils::waitAndCheckForErrors();
}


__global__
void update_bias_kernel(const float* derivative, shape* derivative_shape, float* bias, float learning_rate)
{
    __shared__ float bias_shared[biasThreadPerBlock];
    const unsigned int tr_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int batches = derivative_shape->batches;
    const unsigned int offset = derivative_shape->area();
    const unsigned int batch_offset = derivative_shape->volume();
    bias_shared[threadIdx.x] = 0.0f;

    if (tr_index >= derivative_shape->area())
        return;

    for (int b = 0; b < batches; b++)
    {
        bias_shared[threadIdx.x] += derivative[b * batch_offset + blockIdx.y * offset + tr_index];
    }
    __syncthreads();

    int loop = biasThreadPerBlock;

    while (loop > 1)
    {
        loop = loop / 2;
        if (threadIdx.x < loop)
        {
            bias_shared[threadIdx.x] += bias_shared[threadIdx.x + loop];
        }
        else
            return;
        __syncthreads();
    }
    
    if (threadIdx.x == 0)
    {
        bias_shared[0] = -(learning_rate * (bias_shared[0] / (derivative_shape->area()  * batches)));
        atomicAdd(&bias[blockIdx.y], bias_shared[0]);
    }
}


void update_bias(const float* derivative, shape derivative_shape, float* bias, float learning_rate)
{
    unsigned int num_blocks = ((derivative_shape.area() + biasThreadPerBlock - 1) / biasThreadPerBlock);
    dim3 blocks(num_blocks, derivative_shape.depth);
    utils::device_struct<shape> device_input_shape(derivative_shape);

    update_bias_kernel << <blocks, biasThreadPerBlock >> > (derivative, device_input_shape.get(), bias, learning_rate);

    utils::waitAndCheckForErrors();
}