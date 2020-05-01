#pragma once
#include <device_memory.h>
#include <shape.h>

struct softmax_activation
{
	cuVector<float> exponents;
	cuVector<float> exponents_sum;

	void init(shape input_shape);
	void calc_output(const float* input, float* output);
	void calc_derivatives();
};