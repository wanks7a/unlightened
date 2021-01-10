#include <cuda_vector_norm.h>
#include <cudnn_helpers.h>

float cuda_vector_norm(float* input, size_t size)
{
	cudnn_mul_tensor mul_tensor;
	device_vector<cuda_device, float> input_copy;
	input_copy.memcpy(input, size);
	device_vector<cuda_device, float> vector_norm;
	vector_norm.resize(1, 0.0f);
	if (!mul_tensor.create())
		throw("cudnn_mul_tensor.create() failed");
	if (!mul_tensor.op(input_copy.data(), input_copy.size(), input_copy.data(), input_copy.size(), input_copy.data(), input_copy.size()))
		throw("cudnn_mul_tensor.op() failed");
	cudnn_reduce_add_tensor reduce_op;
	if (!reduce_op.create(size))
		throw("cudnn_reduce_add_tensor.create() failed");
	if(!reduce_op.reduce(input_copy.data(), vector_norm.data()))
		throw("cudnn_reduce_add_tensor.reduce() failed");
	return sqrt(vector_norm[0]);
}

bool cuda_scale_vector(float* ptr, size_t size, float scale)
{
	cudnn_mul_tensor mul_tensor;
	if (!mul_tensor.create())
		return false;
	device_vector<cuda_device, float> scale_vector;
	scale_vector.resize(1, scale);
	return mul_tensor.op(ptr, size, scale_vector.data(), scale_vector.size(), ptr, size);
}