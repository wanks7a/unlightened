#include <gtest/gtest.h>
#include <optimizer_momentum.h>
#include <tests_objects.h>

TEST(momentum_cuda_tests, test1)
{
	device_vector<cuda_device, float> w_props, d_props;
	w_props.set_data({ 1, 1, 1, 1, 1 });
	d_props.set_data({ 10, 10, 10, 10, 10 });
	test_layer t;
	t.set_learning_rate(1.0f);
	t.w_props.size = w_props.size();
	t.w_props.ptr = w_props.data();
	t.d_props.size = d_props.size();
	t.d_props.ptr = d_props.data();
	momentum_optimizer opt(0.0f);
	opt.init(&t);
	opt.update(&t);
	auto res = w_props.to_vector();
	std::vector<float> expected = { -9, -9 , -9, -9, -9 };
	compare_vectors(res, expected);
	opt.update(&t);
	res = w_props.to_vector();
	expected = { -19, -19 , -19, -19, -19 };
	compare_vectors(res, expected);
}