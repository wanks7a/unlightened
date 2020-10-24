#include <gtest/gtest.h>
#include <generic_stream.h>
#include <vector>
#include <binary_serialization.h>
#include <LinearLayer.h>
#include <tests_objects.h>

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