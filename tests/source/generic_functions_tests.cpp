#include <gtest/gtest.h>
#include <device_memory.h>
#include <generic_functions.h>
#include <activation_layer.h>
#include <tests_objects.h>

TEST(generic_functions, sum_all_v1)
{
	cuVector<float> input;
	EXPECT_TRUE(input.resize(2049, 1.0f));
	cuVector<float> out;
	out.resize(1, 0.0f);
	shape sh(2049);
	sum_all_values(sh, input.get(), out.get());
	float expected = 2049;
	auto result = out.to_vector();
	EXPECT_EQ(expected, result[0]);
}

TEST(generic_functions, sum_all_v2)
{
	cuVector<float> input;
	EXPECT_TRUE(input.resize(513, 1.0f));
	cuVector<float> out;
	out.resize(1, 0.0f);
	shape sh(513);
	sum_all_values(sh, input.get(), out.get());
	float expected = 513;
	auto result = out.to_vector();
	EXPECT_EQ(expected, result[0]);
}


TEST(generic_functions, sum_all_v3)
{
	cuVector<float> input;
	EXPECT_TRUE(input.resize(2049 * 2, 1.0f));
	cuVector<float> out;
	out.resize(2, 0.0f);
	shape sh(2049,1,1,2);
	sum_all_values(sh, input.get(), out.get());
	float expected = 2049;
	auto result = out.to_vector();
	EXPECT_EQ(expected, result[0]);
	EXPECT_EQ(expected, result[1]);
}

TEST(generic_functions, softmax_v1)
{
	shape input_sh(3);
	test_layer  test;
	test.init_base(input_sh);
	test.set_output_shape(input_sh);
	EXPECT_TRUE(test.output.setValues({
		1, 3, 5
		}));

	activation_layer a_layer(activation_layer::activation_function::Softmax);
	a_layer.init_base(input_sh);
	a_layer.forward_pass(&test);
	std::vector<float> result = a_layer.get_native_output();
	float val = 0.0f;
	for (size_t i = 0; i < result.size(); i++)
	{
		val += result[i];
	}
	EXPECT_FLOAT_EQ(val, 1.0f);

	test_layer backprop_l;
	backprop_l.init_base(input_sh);
	backprop_l.set_output_shape(input_sh);
	EXPECT_TRUE(backprop_l.output.setValues({
		1, 1, 1
		}));
	a_layer.backprop(&backprop_l);
	result = a_layer.get_native_derivative();
	EXPECT_EQ(result.size(), 3);
}

TEST(generic_functions, softmax_v2)
{
	shape input_sh(3,1,1,2);
	test_layer  test;
	test.init_base(input_sh);
	test.set_output_shape(input_sh);
	EXPECT_TRUE(test.output.setValues({
		1, 3, 5,
		5, 3, 1,
		}));

	activation_layer a_layer(activation_layer::activation_function::Softmax);
	a_layer.init_base(input_sh);
	a_layer.forward_pass(&test);
	std::vector<float> result = a_layer.get_native_output();
	EXPECT_EQ(result.size(), 6);
	float val = 0.0f;
	for (size_t i = 0; i < 3; i++)
	{
		val += result[i];
	}
	EXPECT_FLOAT_EQ(val, 1.0f);
	val = 0.0f;
	for (size_t i = 3; i < 6; i++)
	{
		val += result[i];
	}
	EXPECT_FLOAT_EQ(val, 1.0f);
	test_layer backprop_l;
	backprop_l.init_base(input_sh);
	backprop_l.set_output_shape(input_sh);
	EXPECT_TRUE(backprop_l.output.setValues({
		1, 1, 1,
		1, 1, 1,
		}));
	a_layer.backprop(&backprop_l);
	result = a_layer.get_native_derivative();
	EXPECT_EQ(result.size(), 6);
}