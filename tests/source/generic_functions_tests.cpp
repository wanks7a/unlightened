#include <gtest/gtest.h>
#include <device_memory.h>
#include <generic_functions.h>

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