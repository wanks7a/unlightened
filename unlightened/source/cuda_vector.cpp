#include <cuda_vector.h>
#include <cudnn_helpers.h>
#include <generic_functions.h>

void cuda_floats::operator+=(float val)
{
	vector_add(this->data(), this->size(), val);
}

void cuda_floats::operator*=(float val)
{
	vector_scale(this->data(), this->size(), val);
}

void cuda_floats::operator-=(float val)
{
	this->operator+=(-val);
}

void cuda_floats::operator/=(float val)
{
	this->operator*=(1.0f / val);
}

void cuda_floats::operator+=(const cuda_floats& vals)
{
	if (this->size() == vals.size())
	{
		vector_add(this->data(), vals.data(), this->size());
	}
}

void cuda_floats::operator*=(const cuda_floats& vals)
{
	if (this->size() == vals.size())
	{
		vector_mul(this->data(), vals.data(), this->size());
	}
}

void cuda_floats::operator-=(const cuda_floats& vals)
{
	cudnn_add_tensor tensor;
	tensor.create();
	tensor.op(this->data(), this->size(), vals.data(), vals.size(), this->data(), this->size(), 1.0f, -1.0f);
}

cuda_floats cuda_floats::operator+(float val) const
{
	cuda_floats result;
	result.memcpy(this->data(), this->size());
	result += val;
	return result;
}

cuda_floats cuda_floats::operator*(float val) const
{
	cuda_floats result;
	result.memcpy(this->data(), this->size());
	result *= val;
	return result;
}

cuda_floats cuda_floats::operator-(float val) const
{
	cuda_floats result;
	result.memcpy(this->data(), this->size());
	result -= val;
	return result;
}

cuda_floats cuda_floats::operator/(float val) const
{
	cuda_floats result;
	result.memcpy(this->data(), this->size());
	result *=  (1 / val);
	return result;
}

cuda_floats cuda_floats::operator+(const cuda_floats& vals) const
{
	cuda_floats result;
	result.memcpy(this->data(), this->size());
	result += vals;
	return result;
}

cuda_floats cuda_floats::operator*(const cuda_floats& vals) const
{
	cuda_floats result;
	result.memcpy(this->data(), this->size());
	result *= vals;
	return result;
}

cuda_floats cuda_floats::operator-(const cuda_floats& vals) const
{
	cuda_floats result;
	result.memcpy(this->data(), this->size());
	result -= vals;
	return result;
}

void cuda_floats::sqrt()
{
	vector_sqrt(this->data(), this->size());
}

void cuda_floats::operator/=(const cuda_floats& vals)
{
	if (this->size() != vals.size())
		throw("size doesnt match");
	vector_divide(this->data(), vals.data(), this->size());
}

cuda_floats cuda_floats::operator/(const cuda_floats& vals)
{
	cuda_floats result;
	result.memcpy(this->data(), this->size());
	result /= vals;
	return result;
}