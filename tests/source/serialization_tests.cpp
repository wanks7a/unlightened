#include <gtest/gtest.h>
#include <generic_stream.h>
#include <vector>
#include <binary_serialization.h>
#include <LinearLayer.h>
#include <tests_objects.h>
#include <LinearLayerGPU.h>
#include <activation_layer.h>

struct test_stream : public generic_stream
{
	std::vector<char> s;
	size_t write(const char* ptr, size_t bytes) override
	{
		for (size_t i = 0; i < bytes; i++)
		{
			s.emplace_back(ptr[i]);
		}
		return bytes;
	}

	size_t read(char* buff, size_t bytes) override
	{
		size_t result = peek(buff, bytes);
		s.erase(s.begin(), s.begin() + result);
		return result;
	}

	size_t peek(char* buff, size_t bytes) const override
	{
		size_t result;
		if (s.size() < bytes)
		{
			result = s.size();
		}
		else
		{
			result = bytes;
		}

		for (size_t i = 0; i < result; i++)
		{
			buff[i] = s[i];
		}

		return result;
	}
};

TEST(serialization_tests, dense_layer_serialization)
{
	test_layer t;
	t.set_output_shape(shape(4));
	t.output.setValues({
		1,2,3,4
	});
	binary_serialization ser(std::make_shared<test_stream>());
	std::shared_ptr<Layer> d(new dense_layer(25));
	d->init_base(t.get_shape());
	d->forward_pass(&t);
	auto real_values = d->get_native_output();
	ser.serialize(*d);
	auto expected_layer = ser.deserialize_layer();
	expected_layer->forward_pass(&t);
	auto expected_values = expected_layer->get_native_output();
	EXPECT_EQ(real_values.size(), expected_values.size());
	for (size_t i = 0; i < real_values.size(); i++)
	{
		EXPECT_EQ(real_values[i], expected_values[i]);
	}
}

TEST(serialization_tests, dense_layer_gpu_serialization)
{
	test_layer input_l;
	input_l.set_output_shape(shape(4));
	input_l.output.setValues({
		1,2,3,4
		});
	test_layer backprop_l;
	backprop_l.set_output_shape(shape(25));
	backprop_l.output.setValues({
		1,2,3,4,5,
		1,2,3,4,5,
		1,2,3,4,5,
		1,2,3,4,5,
		1,2,3,4,5,
		});
	binary_serialization ser(std::make_shared<test_stream>());
	std::shared_ptr<Layer> dense_l(new dense_gpu(25));
	dense_l->init_base(input_l.get_shape());

	// forward pass calc
	dense_l->forward_pass(&input_l);
	auto real_values_forward = dense_l->get_native_output();

	// serialize before backprop 
	ser.serialize(*dense_l);
	auto expected_layer = ser.deserialize_layer();

	// backprop calc
	dense_l->backprop(&backprop_l);
	auto real_values_backprop = dense_l->get_native_derivative();
	
	// test results
	expected_layer->forward_pass(&input_l);
	auto expected_values_forward = expected_layer->get_native_output();

	expected_layer->backprop(&backprop_l);
	auto expected_values_backprop = expected_layer->get_native_derivative();

	EXPECT_EQ(real_values_forward.size(), expected_values_forward.size());
	for (size_t i = 0; i < real_values_forward.size(); i++)
	{
		EXPECT_EQ(real_values_forward[i], expected_values_forward[i]);
	}

	EXPECT_EQ(real_values_backprop.size(), expected_values_backprop.size());
	for (size_t i = 0; i < real_values_backprop.size(); i++)
	{
		EXPECT_EQ(real_values_backprop[i], expected_values_backprop[i]);
	}
}

TEST(serialization_tests, activation_layer_serialization)
{
	test_layer input_l;
	input_l.set_output_shape(shape(4));
	input_l.output.setValues({
		1,2,3,4
		});
	test_layer backprop_l;
	backprop_l.set_output_shape(shape(4));
	backprop_l.output.setValues({
		1,2,3,4
		});
	binary_serialization ser(std::make_shared<test_stream>());
	std::shared_ptr<Layer> test_layer(new activation_layer(activation_layer::activation_function::Sigmoid));
	test_layer->init_base(input_l.get_shape());

	// serialize before backprop 
	ser.serialize(*test_layer);
	auto expected_layer = ser.deserialize_layer();

	// forward pass calc
	test_layer->forward_pass(&input_l);
	auto real_values_forward = test_layer->get_native_output();

	// backprop calc
	test_layer->backprop(&backprop_l);
	auto real_values_backprop = test_layer->get_native_derivative();

	// test results
	expected_layer->forward_pass(&input_l);
	auto expected_values_forward = expected_layer->get_native_output();

	expected_layer->backprop(&backprop_l);
	auto expected_values_backprop = expected_layer->get_native_derivative();

	EXPECT_EQ(real_values_forward.size(), expected_values_forward.size());
	for (size_t i = 0; i < real_values_forward.size(); i++)
	{
		EXPECT_EQ(real_values_forward[i], expected_values_forward[i]);
	}

	EXPECT_EQ(real_values_backprop.size(), expected_values_backprop.size());
	for (size_t i = 0; i < real_values_backprop.size(); i++)
	{
		EXPECT_EQ(real_values_backprop[i], expected_values_backprop[i]);
	}
}