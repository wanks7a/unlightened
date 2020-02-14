#pragma once
#include <GpuMemory.h>
#include <vector>
#include <shape.h>
#include <random>

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

class filter_conv2d
{
    filter_options options;
    shape input_shape;
    shape output_shape;
    cuVector<float> weights;

    unsigned int calc_output_dimension(size_t input_dim, size_t filter_dim, unsigned char stride, size_t padding)
    {
        return static_cast<unsigned int>((input_dim - filter_dim + 2 * padding) / stride + 1);
    }

public:
    filter_conv2d() : options(1,1)
    {
    }

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
        if (weights.size() == 0)
        {
            std::default_random_engine generator;
            std::uniform_real_distribution<float> distribution(0.f, 1.0f);
            std::vector<float> input;
            for (size_t i = 0; i < options.w * options.h * options.channels; i++)
            {
                input.emplace_back(distribution(generator));
            }
        }
        return true;
    }

    cuVector<float>& get_weights()
    {
        return weights;
    }

    shape get_output_shape()
    {
        return output_shape;
    }
};