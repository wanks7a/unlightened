#pragma once
#include <GpuMemory.h>
#include <vector>

struct shape
{
    unsigned int width;
    unsigned int height;
    unsigned int depth;

    shape(unsigned int x = 1, unsigned int  y = 1, unsigned int  z = 1) : width(x), height(y), depth(z){}

    shape(const shape& sh) = default;
};

struct filter_options
{
    size_t w;
    size_t h;
    size_t channels;
    unsigned char stride;
    bool zeropadding;
    filter_options(size_t width, size_t height) :w(width), h(height), channels(1), stride(1), zeropadding(true)
    {
    }
};

void filter_forwardPass(const float* input, shape input_shape, const float* weights, float* output, shape output_shape, unsigned int filter_size);

class filter_conv2d
{
    filter_options options;
    shape input_shape;
    shape output_shape;

    unsigned int calc_output_dimension(size_t input_dim, size_t filter_dim, unsigned char stride, size_t padding)
    {
        return static_cast<unsigned int>((input_dim - filter_dim + 2 * padding) / stride + 1);
    }

public:
    filter_conv2d(filter_options opt) : options(opt)
    {
    }

    bool init(shape input)
    {
        input_shape = input;
        size_t padding = 0;
        if (options.zeropadding)
        {
            padding = (options.w - 1) / 2;
        }
        output_shape.width = calc_output_dimension(input_shape.width, options.w, options.stride, padding);
        output_shape.height = calc_output_dimension(input_shape.height, options.h, options.stride, padding);
        return true;
    }

    shape get_output_shape()
    {
        return output_shape;
    }
};