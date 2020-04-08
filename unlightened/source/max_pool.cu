#include <max_pool.h>
#include <GpuUtils.h>

template <int FILTER_SIZE> __global__
void k_max_pool(const float* input, shape* input_shape, float* output, shape* output_shape, char* mask)
{
    const unsigned int tr_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (tr_index >= output_shape->volume())
        return;
    const unsigned int depth = tr_index / output_shape->area();
    unsigned int row = tr_index / output_shape->width;
    const unsigned int col = tr_index - (row * output_shape->width);
    const unsigned int input_width = input_shape->width;
    const unsigned int input_height = input_shape->height;
    const unsigned int depth_offset_input = depth * input_shape->area();
    const unsigned int batch_offset_input = blockIdx.y * input_shape->volume();
    const unsigned int batch_offset_output = blockIdx.y * output_shape->volume();

    float result = 0.0f;
    char mask_temp = 0;
    row = row - depth * output_shape->height;
    for (unsigned int i = 0; i < FILTER_SIZE; i++)
    {
        unsigned int input_row = row * FILTER_SIZE + i;
        for (unsigned int j = 0; j < FILTER_SIZE; j++)
        {
            unsigned int input_col = col * FILTER_SIZE + j;
            if (input_row >= input_height || input_col >= input_width)
            {
            }
            else
            {
                if (result < input[batch_offset_input + depth_offset_input + input_row * input_width + input_col])
                {
                    result = input[batch_offset_input + depth_offset_input + input_row * input_width + input_col];
                    mask_temp = i * FILTER_SIZE + j;
                }
            }
        }
    }
    output[batch_offset_output + tr_index] = result;
    mask[batch_offset_output + tr_index] = mask_temp;
}

void max_pooling(const float* input,const shape& input_shape, float* output, const shape& output_shape, char* mask, int filter_size)
{
    unsigned int threads_per_block = 256;
    unsigned int num_blocks = ((output_shape.volume() + threads_per_block - 1) / threads_per_block);
    dim3 blocks(num_blocks, input_shape.batches);
    utils::device_struct<shape> device_input_shape(input_shape);
    utils::device_struct<shape> device_output_shape(output_shape);

    switch (filter_size)
    {
        case 2: k_max_pool<2><<<blocks, threads_per_block>>> (input, device_input_shape.get(), output, device_output_shape.get(), mask); break;
        case 3: k_max_pool<3><<<blocks, threads_per_block>>> (input, device_input_shape.get(), output, device_output_shape.get(), mask); break;
        case 4: k_max_pool<4><<<blocks, threads_per_block>>> (input, device_input_shape.get(), output, device_output_shape.get(), mask); break;
        case 5: k_max_pool<5><<<blocks, threads_per_block>>> (input, device_input_shape.get(), output, device_output_shape.get(), mask); break;
        case 6: k_max_pool<6><<<blocks, threads_per_block>>> (input, device_input_shape.get(), output, device_output_shape.get(), mask); break;
        case 7: k_max_pool<7><<<blocks, threads_per_block>>> (input, device_input_shape.get(), output, device_output_shape.get(), mask); break;
        default: break;
    }

    utils::waitAndCheckForErrors();
}

template <int FILTER_SIZE> __global__
void k_max_pool_backprop(const float* input, shape* input_shape, float* output, shape* output_shape, char* mask)
{
    const unsigned int tr_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (tr_index >= input_shape->volume())
        return;
    const unsigned int depth = tr_index / input_shape->area();
    unsigned int row = tr_index / input_shape->width;
    const unsigned int col = tr_index - (row * input_shape->width);
    const unsigned int depth_offset_output = depth * output_shape->area();
    const unsigned int batch_offset_input = blockIdx.y * input_shape->volume();
    const unsigned int batch_offset_output = blockIdx.y * output_shape->volume();
    const unsigned int output_width = output_shape->width;
    const unsigned int output_height = output_shape->height;

    char mask_temp = mask[batch_offset_input + tr_index];
    row = row - depth * input_shape->height;
    for (unsigned int i = 0; i < FILTER_SIZE; i++)
    {
        unsigned int row_output_idx = row * FILTER_SIZE + i;
        for (unsigned int j = 0; j < FILTER_SIZE; j++)
        {
            unsigned int col_input_idx = col * FILTER_SIZE + j;
            if(row_output_idx < output_width && col_input_idx < output_height)
                output[batch_offset_output + depth_offset_output + row_output_idx * output_width + col_input_idx] = 0;
        }
    }
    unsigned int mask_row_idx = mask_temp / FILTER_SIZE;
    unsigned int mask_col_idx = mask_temp - mask_row_idx * FILTER_SIZE;
    output[batch_offset_output + depth_offset_output + (row * FILTER_SIZE + mask_row_idx) * output_width + col * FILTER_SIZE + mask_col_idx] = input[batch_offset_input + tr_index];
}

void max_pooling_backprop(const float* input, const shape& input_shape, float* output, const shape& output_shape, char* mask, int filter_size)
{
    unsigned int threads_per_block = 256;
    unsigned int num_blocks = ((input_shape.volume() + threads_per_block - 1) / threads_per_block);
    dim3 blocks(num_blocks, input_shape.batches);
    utils::device_struct<shape> device_input_shape(input_shape);
    utils::device_struct<shape> device_output_shape(output_shape);

    switch (filter_size)
    {
    case 2: k_max_pool_backprop<2> << <blocks, threads_per_block >> > (input, device_input_shape.get(), output, device_output_shape.get(), mask); break;
    case 3: k_max_pool_backprop<3> << <blocks, threads_per_block >> > (input, device_input_shape.get(), output, device_output_shape.get(), mask); break;
    case 4: k_max_pool_backprop<4> << <blocks, threads_per_block >> > (input, device_input_shape.get(), output, device_output_shape.get(), mask); break;
    case 5: k_max_pool_backprop<5> << <blocks, threads_per_block >> > (input, device_input_shape.get(), output, device_output_shape.get(), mask); break;
    case 6: k_max_pool_backprop<6> << <blocks, threads_per_block >> > (input, device_input_shape.get(), output, device_output_shape.get(), mask); break;
    case 7: k_max_pool_backprop<7> << <blocks, threads_per_block >> > (input, device_input_shape.get(), output, device_output_shape.get(), mask); break;
    default: break;
    }

    utils::waitAndCheckForErrors();
}