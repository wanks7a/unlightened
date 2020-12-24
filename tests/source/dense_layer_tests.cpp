#include <gtest/gtest.h>
#include <LinearLayerGPU.h>
#include <tests_objects.h>

TEST(dense_layer_tests, derivative_to_input_v1)
{
	size_t inputSize = 2;
	shape output_shape(2);
	cuVector<float> weights;
	EXPECT_TRUE(weights.setValues({
		1, 2,
		3, 4
		}));
	cuVector<float> derivativeWRtoInput;
	EXPECT_TRUE(derivativeWRtoInput.setValues({ 0, 0 }));
	cuVector<float> derivateWRtoOutput;
	EXPECT_TRUE(derivateWRtoOutput.setValues({1, 1}));
	calcDerivativeWRtoInput(derivativeWRtoInput.get(), 2, derivateWRtoOutput.get(), output_shape, weights.get());
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

TEST(dense_layer_tests, derivative_to_input_v2)
{
	size_t inputSize = 2;
	shape output_shape(2,1,1,2);
	cuVector<float> weights;
	EXPECT_TRUE(weights.setValues({
		1, 2,
		3, 4,
		}));
	cuVector<float> derivativeWRtoInput;
	EXPECT_TRUE(derivativeWRtoInput.setValues({ 0, 0, 0, 0 }));
	cuVector<float> derivateWRtoOutput;
	EXPECT_TRUE(derivateWRtoOutput.setValues({ 1, 1, 2, 2 }));
	calcDerivativeWRtoInput(derivativeWRtoInput.get(), 2, derivateWRtoOutput.get(), output_shape, weights.get());
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

TEST(dense_layer_tests, derivative_to_input_v3)
{
	size_t inputSize = 2;
	shape output_shape(2, 1, 1, 3);
	cuVector<float> weights;
	EXPECT_TRUE(weights.setValues({
		1, 2,
		3, 4,
		}));
	cuVector<float> derivativeWRtoInput;
	EXPECT_TRUE(derivativeWRtoInput.setValues({ 0, 0, 0, 0, 0, 0 }));
	cuVector<float> derivateWRtoOutput;
	EXPECT_TRUE(derivateWRtoOutput.setValues({ 1, 1, 2, 2, 5, 5 }));
	calcDerivativeWRtoInput(derivativeWRtoInput.get(), 2, derivateWRtoOutput.get(), output_shape, weights.get());
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

TEST(dense_layer_tests, weights_deriv_v1)
{
	cuVector<float> weights;
	cuVector<float> weights_deriv;
	EXPECT_TRUE(weights.setValues({
		1,2,
		3,4,
		5,6
		}));
	weights_deriv.resize(weights.size());
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
	calcWeightsDeriv(weights.get(), weights_deriv.get(), derivativeWRtoOutput.get(), input.get(), inputSize, outputSize, output_shape);
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

TEST(dense_layer_tests, weights_deriv_v2)
{
	cuVector<float> weights;
	cuVector<float> weights_deriv;
	EXPECT_TRUE(weights.setValues({
		1,2,
		3,4
		}));
	weights_deriv.resize(weights.size());
	cuVector<float> derivativeWRtoOutput;
	EXPECT_TRUE(derivativeWRtoOutput.setValues({
		10, 20
		}));
	cuVector<float> input;
	EXPECT_TRUE(input.setValues({
		1, 2
		}));
	size_t inputSize = 2;
	size_t outputSize = 2;
	shape output_shape(2);
	calcWeightsDeriv(weights.get(), weights_deriv.get(), derivativeWRtoOutput.get(), input.get(), inputSize, outputSize, output_shape);
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


TEST(dense_layer_tests, weights_deriv_v3)
{
	cuVector<float> weights;
	cuVector<float> weights_deriv;
	EXPECT_TRUE(weights.setValues({
		1,2,
		3,4
		}));
	weights_deriv.resize(weights.size());
	cuVector<float> derivativeWRtoOutput;
	EXPECT_TRUE(derivativeWRtoOutput.setValues({
		10, 20, 
		2, 2,
		}));
	cuVector<float> input;
	EXPECT_TRUE(input.setValues({
		1, 2,
		1, 1,
		}));
	size_t inputSize = 2;
	size_t outputSize = 2;
	shape output_shape(2, 1, 1, 2);
	calcWeightsDeriv(weights.get(), weights_deriv.get(), derivativeWRtoOutput.get(), input.get(), inputSize, outputSize, output_shape);
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

TEST(dense_layer_tests, bias_update_v1)
{
	cuVector<float> bias;
	cuVector<float> bias_deriv;
	bias.resize(2048, 1.0f);
	bias_deriv.resize(bias.size());
	std::vector<float> deriv_temp;
	deriv_temp.resize(2048);
	for (size_t i = 0; i < 1024; i++)
	{
		deriv_temp[i] = 10.0f;
		deriv_temp[i + 1024] = 20.0f;
	}
	cuVector<float> derivative;
	derivative.setValues(deriv_temp);

	calcBiasDeriv(bias.get(), bias_deriv.get(), derivative.get(), 2048, shape(2048));
	auto result = bias_deriv.to_vector();
	EXPECT_EQ(result.size(), 2048);
	for (size_t i = 0; i < 1024; i++)
	{
		EXPECT_EQ(result[i], 10.0f);
		EXPECT_EQ(result[i + 1024], 20.0f);
	}
}

TEST(dense_layer_tests, bias_update_v1_batched)
{
	shape out(2048, 1, 1, 2);
	cuVector<float> bias;
	cuVector<float> bias_deriv;
	bias.resize(2048, 1.0f);
	bias_deriv.resize(bias.size());
	std::vector<float> deriv_temp;
	deriv_temp.resize(2048 * 2);
	for (size_t i = 0; i < 1024; i++)
	{
		deriv_temp[i] = 5.0f;
		deriv_temp[i + 1024] = 10.0f;
		deriv_temp[i + 2 * 1024] = 15.0f;
		deriv_temp[i + 3 * 1024] = 20.0f;
	}
	cuVector<float> derivative;
	derivative.setValues(deriv_temp);

	calcBiasDeriv(bias.get(), bias_deriv.get(), derivative.get(), 2048, out);
	auto result = bias_deriv.to_vector();
	EXPECT_EQ(result.size(), 2048);
	for (size_t i = 0; i < 1024; i++)
	{
		EXPECT_EQ(result[i], 20.0f);
		EXPECT_EQ(result[i + 1024], 30.0f);
	}
}

TEST(dense_layer_tests, test_forward_backprop_without_bias_v1)
{
	shape input_sh(10);
	test_layer  test;
	test.init_base(input_sh);
	test.set_output_shape(input_sh);
	EXPECT_TRUE(test.output.setValues({
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1
		}));

	dense_gpu dense(3);
	dense.init_base(input_sh);
	std::vector<float> weights(10 * 3, 1.0f);
	EXPECT_TRUE(dense.set_weights(weights));
	dense.forward_pass(&test);
	std::vector<float> result = dense.get_native_output();
	std::vector<float> expected = {
		10, 10, 10
	};
	EXPECT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < expected.size(); i++)
	{
		EXPECT_EQ(result[i], expected[i]);
	}

	test_layer  backprop_test;
	backprop_test.init_base(shape(3));
	backprop_test.set_output_shape(shape(3));
	EXPECT_TRUE(backprop_test.output.setValues({
		1, 1, 1
		}));
	dense.backprop(&backprop_test);
	result = dense.get_native_derivative();
	expected = {
		3, 3, 3, 3, 3,
		3, 3, 3, 3, 3
	};
	EXPECT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < expected.size(); i++)
	{
		EXPECT_EQ(result[i], expected[i]);
	}
}

TEST(dense_layer_tests, test_forward_backprop_without_bias_v2)
{
	shape input_sh(10, 1, 1, 2);
	test_layer  test;
	test.init_base(input_sh);
	test.set_output_shape(input_sh);
	EXPECT_TRUE(test.output.setValues({
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		2, 2, 2, 2, 2,
		2, 2, 2, 2, 2,
		}));

	dense_gpu dense(3);
	dense.init_base(input_sh);
	std::vector<float> weights(10 * 3, 1.0f);
	EXPECT_TRUE(dense.set_weights(weights));
	dense.forward_pass(&test);
	std::vector<float> result = dense.get_native_output();
	std::vector<float> expected = {
		10, 10, 10,
		20, 20, 20,
	};
	EXPECT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < expected.size(); i++)
	{
		EXPECT_EQ(result[i], expected[i]);
	}

	test_layer  backprop_test;
	backprop_test.init_base(shape(3, 1, 1, 2));
	backprop_test.set_output_shape(shape(3, 1, 1, 2));
	EXPECT_TRUE(backprop_test.output.setValues({
		1, 1, 1,
		1, 1, 1,
		}));
	dense.backprop(&backprop_test);
	result = dense.get_native_derivative();
	expected = {
		3, 3, 3, 3, 3,
		3, 3, 3, 3, 3,
		3, 3, 3, 3, 3,
		3, 3, 3, 3, 3
	};
	EXPECT_EQ(result.size(), expected.size());
	for (size_t i = 0; i < expected.size(); i++)
	{
		EXPECT_EQ(result[i], expected[i]);
	}
}

TEST(dense_layer_tests, forward_pass_dense_1024)
{
	std::vector<float> input;
	input.resize(1024 * 1024, 1.0f);

	cuVector<float> inputPtr;
	EXPECT_TRUE(inputPtr.resize(1024, 1.0f));
	cuVector<float> weightsGPU;
	EXPECT_TRUE(weightsGPU.setValues(input));
	cuVector<float> output;
	EXPECT_TRUE(output.resize(1024));
	cuVector<float> bias;
	EXPECT_TRUE(bias.resize(1024, 1.0f));

	std::vector<float> result;

	linearLayerForwardPassGPU(output.get(), weightsGPU.get(), inputPtr.get(), 1024, bias.get(), 1024);
	
	output.getCopy(result);
	for (size_t i = 0; i < result.size(); i++)
	{
		EXPECT_EQ(1025, result[i]);
	}
}

TEST(dense_layer_tests, dense_layer_tests1024_batches2_v1)
{
	shape input_shape(1024, 1, 1, 2);
	cuVector<float> input;
	EXPECT_TRUE(input.resize(input_shape.size(), 1.0f));
	shape output_shape(1024, 1, 1, 2);
	cuVector<float> output;
	EXPECT_TRUE(output.resize(output_shape.size()));
	cuVector<float> weights;
	EXPECT_TRUE(weights.resize(1024 * 1024, 1.0f));
	cuVector<float> bias;
	EXPECT_TRUE(bias.resize(1024, 1.0f));

	std::vector<float> result;
	linearLayerForwardPassGPU(output.get(), weights.get(), input.get(), input_shape, bias.get(),output_shape);
	output.getCopy(result);
	EXPECT_EQ(result.size(), output_shape.size());
	for (size_t i = 0; i < result.size(); i++)
	{
		EXPECT_EQ(1025, result[i]);
	}
}

TEST(dense_layer_tests, dense_layer_tests1024_batches2_v2)
{
	shape input_shape(1024, 1, 1, 2);
	cuVector<float> input;
	std::vector<float> input_temp;
	input_temp.resize(input_shape.size());
	for (size_t i = 0; i < 1024; i++)
	{
		input_temp[i] = 1.0f;
		input_temp[i + 1024] = 2.0f;
	}
	EXPECT_TRUE(input.setValues(input_temp));
	shape output_shape(1024, 1, 1, 2);
	cuVector<float> output;
	EXPECT_TRUE(output.resize(output_shape.size()));
	cuVector<float> weights;
	EXPECT_TRUE(weights.resize(1024 * 1024, 1.0f));
	cuVector<float> bias;
	EXPECT_TRUE(bias.resize(1024, 1.0f));

	std::vector<float> result;
	linearLayerForwardPassGPU(output.get(), weights.get(), input.get(), input_shape, bias.get(), output_shape);
	output.getCopy(result);
	EXPECT_EQ(result.size(), output_shape.size());
	for (size_t i = 0; i < result.size(); i++)
	{
		if (i < 1024)
			EXPECT_EQ(1025, result[i]);
		else
			EXPECT_EQ(2049, result[i]);
	}
}

TEST(dense_layer_tests, dense_layer_generic_v1)
{
	size_t input_size_dense = 10;
	size_t input_size_test_l = 4;
	test_layer test_l;
	test_l.set_output_shape(shape(input_size_test_l, 1, 1, 10));
	std::vector<float> data;
	for (size_t i = 0; i < test_l.get_shape().size(); i++)
	{
		data.emplace_back(i + 1.0f);
	}

	std::vector<float> weights;
	for (size_t i = 0; i <  input_size_dense; i++)
	{
		for (size_t j = 0; j < input_size_test_l; j++)
		{
			weights.emplace_back(i + 1.0f);
		}
	}

	test_l.output.setValues(data);

	dense_gpu d(10);
	d.init_base(test_l.get_shape());
	EXPECT_TRUE(d.set_weights(weights));
	d.forward_pass(&test_l);
	auto result = d.get_native_output();
	std::vector<float> expected;
	for (size_t i = 0; i < 10; i++)
	{
		for (size_t k = 0; k < 10; k++)
		{
			float result = 0.0f;
			for (size_t j = 0; j < 4; j++)
			{
				result += data[i*4 + j];
			}
			result *= weights[k * 4];
			expected.push_back(result);
		}
	}
	EXPECT_EQ(expected.size(), result.size());
	for (size_t i = 0; i < expected.size(); i++)
	{
		EXPECT_EQ(expected[i], result[i]);
	}
}