#include <shape.h>

void max_pool(const float* input, const shape& input_shape, float* output, const shape& output_shape, char* mask, int filter_size);

void max_pool_backprop(const float* input, const shape& input_shape, float* output, const shape& output_shape, char* mask, int filter_size);