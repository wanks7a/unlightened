#pragma once

struct shape
{
    size_t width;
    size_t height;
    size_t depth;

    shape(size_t x = 1, size_t y = 1, size_t z = 1) : width(x), height(y), depth(z) {}

    shape(const shape& sh) = default;

    size_t size() const
    {
        return width * height * depth;
    }
};