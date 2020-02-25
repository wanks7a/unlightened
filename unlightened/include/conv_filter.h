#pragma once
#include <device_memory.h>
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
    size_t num_of_filters;
    filter_options(size_t width, size_t height, size_t number_of_filters) :w(width), h(height), channels(1), stride(1), zeropadding(false), num_of_filters(number_of_filters)
    {
    }
};

class filter_conv2d
{
    filter_options options;
    shape input_shape;
    shape output_shape;
    cuVector<float> weights;
    cuVector<float> weights_derivative;
    size_t padding = 0;

    unsigned int calc_output_dimension(size_t input_dim, size_t filter_dim, unsigned char stride, size_t padding)
    {
        return static_cast<unsigned int>((input_dim - filter_dim + 2 * padding) / stride + 1);
    }

public:
    filter_conv2d() : options(1,1,1)
    {
    }

    filter_conv2d(filter_options opt) : options(opt)
    {
    }

    bool init(shape input)
    {
        if (input.depth != options.channels)
            options.channels = input.depth;
        input_shape = input;
        padding = 0;
        if (options.zeropadding)
        {
            padding = (options.w - 1) / 2;
        }
        output_shape.width = calc_output_dimension(input_shape.width, options.w, options.stride, padding);
        output_shape.height = calc_output_dimension(input_shape.height, options.h, options.stride, padding);
        output_shape.depth = options.num_of_filters;
        output_shape.batches = input_shape.batches;

        if (weights.size() == 0)
        {
            std::default_random_engine generator;
            std::uniform_real_distribution<float> distribution(0.f, 1.0f);
            std::vector<float> weights_values;
            for (size_t i = 0; i < options.w * options.h * options.channels * options.num_of_filters; i++)
            {
                weights_values.emplace_back(distribution(generator));
            }
            weights.setValues(weights_values);
        }
        weights_derivative.resize(options.w * options.h * options.channels * options.num_of_filters * output_shape.batches);
        return true;
    }

    cuVector<float>& get_weights()
    {
        return weights;
    }

    cuVector<float>& get_weights_derivative()
    {
        return weights_derivative;
    }

    shape get_output_shape()
    {
        return output_shape;
    }

    inline size_t size() const
    {
        return options.num_of_filters;
    }

    float* operator[](size_t filter_index)
    {
        if (filter_index >= options.num_of_filters)
        {
            std::cout << "wrong filter index" << std::endl;
            return nullptr;
        }
        return weights.get() + filter_index * options.w * options.h * options.channels;
    }

    const filter_options& get_options() const
    {
        return options;
    }

    void set_options(filter_options opt)
    {
        options = opt;
    }
    
    size_t get_padding() const
    {
        return padding;
    }

    shape get_weights_derivative_shape() const
    {
        shape derivatives_shape;
        derivatives_shape.width = options.w;
        derivatives_shape.height = options.h;
        derivatives_shape.depth = options.channels;
        return derivatives_shape;
    }
};