#include <gtest/gtest.h>
#include <LinearLayerGPU.h>

TEST(dense_layer, derivative_to_input_v1)
{
	size_t inputSize = 2;
	shape output_shape(3);
	cuVector<float> weights;
	EXPECT_TRUE(weights.setValues({
		1, 2,
		3, 4
		}));
	cuVector<float> derivativeWRtoInput;
	EXPECT_TRUE(derivativeWRtoInput.setValues({ 0, 0 }));
	cuVector<float> derivateWRtoOutput;
	EXPECT_TRUE(derivateWRtoOutput.setValues({1, 1, 1}));
	calcDerivativeWRtoInput(derivativeWRtoInput.get(), 2, derivateWRtoOutput.get(), output_shape, weights.get(), true);
	std::vector<float> result = derivativeWRtoInput.to_vector();
	std::vector<float> expected = {
		4, 6
	};
	EXPECT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < expected.size(); i++)
	{
		EXPECT_EQ(result[i], expected[i]);
	}
}

TEST(dense_layer, derivative_to_input_v2)
{
	size_t inputSize = 2;
	shape output_shape(3,1,1,2);
	cuVector<float> weights;
	EXPECT_TRUE(weights.setValues({
		1, 2,
		3, 4,
		}));
	cuVector<float> derivativeWRtoInput;
	EXPECT_TRUE(derivativeWRtoInput.setValues({ 0, 0, 0, 0 }));
	cuVector<float> derivateWRtoOutput;
	EXPECT_TRUE(derivateWRtoOutput.setValues({ 1, 1, 1, 2, 2, 2 }));
	calcDerivativeWRtoInput(derivativeWRtoInput.get(), 2, derivateWRtoOutput.get(), output_shape, weights.get(), true);
	std::vector<float> result = derivativeWRtoInput.to_vector();
	std::vector<float> expected = {
		4, 6, 8, 12
	};
	EXPECT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < expected.size(); i++)
	{
		EXPECT_EQ(result[i], expected[i]);
	}
}

TEST(dense_layer, derivative_to_input_v3)
{
	size_t inputSize = 2;
	shape output_shape(3, 1, 1, 3);
	cuVector<float> weights;
	EXPECT_TRUE(weights.setValues({
		1, 2,
		3, 4,
		}));
	cuVector<float> derivativeWRtoInput;
	EXPECT_TRUE(derivativeWRtoInput.setValues({ 0, 0, 0, 0, 0, 0 }));
	cuVector<float> derivateWRtoOutput;
	EXPECT_TRUE(derivateWRtoOutput.setValues({ 1, 1, 1, 2, 2, 2, 5, 5, 5 }));
	calcDerivativeWRtoInput(derivativeWRtoInput.get(), 2, derivateWRtoOutput.get(), output_shape, weights.get(), true);
	std::vector<float> result = derivativeWRtoInput.to_vector();
	std::vector<float> expected = {
		4, 6, 8, 12, 20, 30
	};
	EXPECT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < expected.size(); i++)
	{
		result[i] = expected[i];
	}
}

TEST(dense_layer, weights_update_v1)
{
	cuVector<float> weights;
	EXPECT_TRUE(weights.setValues({
		1,2,
		3,4,
		5,6
		}));
	cuVector<float> derivativeWRtoOutput;
	EXPECT_TRUE(derivativeWRtoOutput.setValues({
		10, 20, 30
		}));
	cuVector<float> input;
	EXPECT_TRUE(input.setValues({
		1, 2
		}));
	size_t inputSize = 2;
	size_t outputSize = 3;
	shape output_shape(3);
	updateWeightsAndBias(weights.get(), derivativeWRtoOutput.get(), input.get(), inputSize, outputSize, output_shape, 1.0f);
	std::vector<float> result = weights.to_vector();
	std::vector<float> expected =
	{
		-9, -18,
		-17, -36,
		-25, -54
	};
	EXPECT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < expected.size(); i++)
	{
		result[i] = expected[i];
	}
}

TEST(dense_layer, weights_update_v2)
{
	cuVector<float> weights;
	EXPECT_TRUE(weights.setValues({
		1,2,
		3,4
		}));
	cuVector<float> derivativeWRtoOutput;
	EXPECT_TRUE(derivativeWRtoOutput.setValues({
		10, 20, 9999
		}));
	cuVector<float> input;
	EXPECT_TRUE(input.setValues({
		1, 2
		}));
	size_t inputSize = 2;
	size_t outputSize = 3;
	shape output_shape(3);
	updateWeightsAndBias(weights.get(), derivativeWRtoOutput.get(), input.get(), inputSize, outputSize - 1, output_shape, 1.0f);
	std::vector<float> result = weights.to_vector();
	std::vector<float> expected =
	{
		-9, -18,
		-17, -36
	};
	EXPECT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < expected.size(); i++)
	{
		result[i] = expected[i];
	}
}

TEST(dense_layer, weights_update_v3)
{
	cuVector<float> weights;
	EXPECT_TRUE(weights.setValues({
		1,2,
		3,4
		}));
	cuVector<float> derivativeWRtoOutput;
	EXPECT_TRUE(derivativeWRtoOutput.setValues({
		10, 20, 9999,
		2, 2, 0,
		}));
	cuVector<float> input;
	EXPECT_TRUE(input.setValues({
		1, 2,
		1, 1,
		}));
	size_t inputSize = 2;
	size_t outputSize = 3;
	shape output_shape(3, 1, 1, 2);
	updateWeightsAndBias(weights.get(), derivativeWRtoOutput.get(), input.get(), inputSize, outputSize - 1, output_shape, 1.0f);
	std::vector<float> result = weights.to_vector();
	std::vector<float> expected =
	{
		-5, -9,
		-8, -17
	};
	EXPECT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < expected.size(); i++)
	{
		result[i] = expected[i];
	}
}