#pragma once
#include <cuda_runtime.h>

struct shape
{
    size_t width;
    size_t height;
    size_t depth;
    size_t batches;

    shape(size_t x = 1, size_t y = 1, size_t z = 1, size_t b = 1) : width(x), height(y), depth(z), batches(b) {}

    shape(const shape& sh) = default;

    __device__
    __host__
    __forceinline__
    size_t size() const
    {
        return width * height * depth * batches;
    }

    __device__
    __host__
    __forceinline__
    size_t area() const
    {
        return width * height;
    }

    __device__
    __host__
    __forceinline__
    size_t volume() const
    {
        return width * height * depth;
    }

    __device__
    __host__
    __forceinline__
    bool operator==(const shape& sh)
    {
        return width == sh.width && height == sh.height && depth == sh.depth && batches == sh.batches;
    }
};