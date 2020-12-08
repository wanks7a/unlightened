#include <gtest/gtest.h>
#include <generic_stream.h>
#include <vector>
#include <binary_serialization.h>
#include <LinearLayer.h>
#include <tests_objects.h>
#include <LinearLayerGPU.h>
#include <activation_layer.h>
#include <max_pool.h>
#include <conv2d_cudnn.h>
#include <NeuralNet.h>
#include <reshape_layer.h>
#include <conv_transpose.h>

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

TEST(serialization_tests, max_pool_serialization)
{
	test_layer input_l;
	input_l.set_output_shape(shape(4,4));
	input_l.output.setValues({
		1,2,3,4,
		1,2,3,4,
		1,2,3,4,
		1,2,3,4,
		});
	test_layer backprop_l;
	backprop_l.set_output_shape(shape(2,2));
	backprop_l.output.setValues({
		1,2,3,4
		});
	binary_serialization ser(std::make_shared<test_stream>());
	std::shared_ptr<Layer> test_layer(new max_pool(2));
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

TEST(serialization_tests, conv2d_gpu_serialization)
{
	test_layer input_l;
	input_l.set_output_shape(shape(4, 4));
	input_l.output.setValues({
		1,2,3,4,
		1,2,3,4,
		1,2,3,4,
		1,2,3,4,
		});

	test_layer backprop_l;
	backprop_l.set_output_shape(shape(3, 3, 10));
	std::vector<float> vals;
	vals.resize(3 * 3 * 10, 1.0f);
	backprop_l.output.setValues(vals);

	binary_serialization ser(std::make_shared<test_stream>());
	std::shared_ptr<Layer> test_layer(new conv2d_cudnn(2, 10));
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

TEST(serialization_tests, neural_net_serialization_test)
{
	binary_serialization ser(std::make_shared<test_stream>());

	NeuralNet n(shape(10,10));
	n.addLayer(new conv2d_cudnn(2, 3, false));
	n.addLayer(new activation_layer(activation_layer::activation_function::Softmax));
	n.addLayer(new dense_gpu(10));
	n.addLayer(new max_pool(2));
	
	n.serialize(ser);
	auto saved_model = ser.deserialize_model();

	// test results
	std::vector<float> input;
	for (size_t i = 0; i < 10 * 10; i++)
	{
		input.push_back(i);
	}

	n.getInputLayer().set_input(input);
	n.predict();
	auto expected_results = n.loss_layer().get_native_output();

	saved_model->getInputLayer().set_input(input);
	saved_model->predict();
	auto results = saved_model->loss_layer().get_native_output();

	// forward pass check
	EXPECT_EQ(expected_results.size(), results.size());
	for (size_t i = 0; i < expected_results.size(); i++)
	{
		EXPECT_EQ(expected_results[i], results[i]);
	}
	
	input.clear();
	for (size_t i = 0; i < n.loss_layer().get_shape().size(); i++)
	{
		input.push_back(i);
	}

	n.loss_layer().setObservedValue(input);
	n.backprop();
	expected_results = n[0]->get_native_derivative();

	saved_model->loss_layer().setObservedValue(input);
	saved_model->backprop();
	results = (*saved_model)[0]->get_native_derivative();

	// backprop check
	EXPECT_EQ(expected_results.size(), results.size());
	for (size_t i = 0; i < expected_results.size(); i++)
	{
		if (std::isnan(expected_results[i]) && std::isnan(results[i]))
		{
		}
		else
			EXPECT_EQ(expected_results[i], results[i]);
	}
}


TEST(serialization_tests, reshape_layer_test_serialization)
{
	binary_serialization ser(std::make_shared<test_stream>());
	std::shared_ptr<Layer> test_layer(new reshape_layer(shape(20,20)));
	test_layer->init_base(shape(20 * 20));
	ser.serialize(*test_layer);
	auto layer = ser.deserialize_layer();
	EXPECT_EQ(test_layer->get_shape().width, layer->get_shape().width);
	EXPECT_EQ(test_layer->get_shape().height, layer->get_shape().height);
	EXPECT_EQ(test_layer->get_shape().depth, layer->get_shape().depth);
	EXPECT_EQ(test_layer->get_shape().batches, layer->get_shape().batches);
}

TEST(serialization_tests, conv2d_transposed_serialization)
{
	test_layer input_l;
	input_l.set_output_shape(shape(4, 4));
	input_l.output.setValues({
		1,2,3,4,
		1,2,3,4,
		1,2,3,4,
		1,2,3,4,
		});
	test_layer backprop_l;
	
	binary_serialization ser(std::make_shared<test_stream>());
	std::shared_ptr<Layer> test_layer(new conv2d_transposed(4, 5, 2, conv2d_transposed::padding::VALID));
	test_layer->init_base(input_l.get_shape());

	backprop_l.set_output_shape(test_layer->get_shape());
	std::vector<float> values;
	for (size_t i = 0; i < test_layer->get_shape().size(); i++)
	{
		values.push_back(i);
	}
	backprop_l.output.setValues(values);

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