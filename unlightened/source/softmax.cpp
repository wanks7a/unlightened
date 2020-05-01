#include <softmax.h>

void softmax_activation::init(shape input_shape)
{
	exponents.resize(input_shape.size(), 0.0f);
	exponents_sum.resize(input_shape.batches, 0.0f);
}