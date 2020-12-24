#include <gtest/gtest.h>
#include <cuda_device.h>
#include <device_vector.h>
#include <tests_objects.h>

TEST(cuda_device_test, test1)
{
	std::vector<float> data = { 1, 2, 3 };
	device_vector<cuda_device, float> test1;
	test1.set_data(data);
	auto res = test1.to_vector();
	compare_vectors(data, res);
}

TEST(cuda_device_test, test2)
{
	std::vector<float> data = { 1, 2, 3 };
	device_vector<cuda_device, float> test1;
	test1.set_data(data);
	auto device_data_copy = test1.copy();
	auto res = device_data_copy.to_vector();
	compare_vectors(data, res);
}