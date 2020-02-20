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
    int mask_temp = 0;
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

void max_pool(const float* input,const shape& input_shape, float* output, const shape& output_shape, char* mask, int filter_size)
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