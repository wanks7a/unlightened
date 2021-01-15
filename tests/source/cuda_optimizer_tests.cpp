#include <gtest/gtest.h>
#include <optimizer_momentum.h>
#include <tests_objects.h>
#include <cuda_vector_norm.h>
#include <adam_optimizer.h>
#include <cuda_vector.h>
#include <generic_functions.h>

TEST(momentum_cuda_tests, test1)
{
	device_vector<cuda_device, float> w_props, d_props, bias_props, bias_deriv;
	w_props.set_data({ 1, 1, 1, 1, 1 });
	d_props.set_data({ 10, 10, 10, 10, 10 });
	bias_props.set_data({ 1, 1, 1, 1, 1 });
	bias_deriv.set_data({ 2, 2, 2, 2, 2 });
	test_layer t;
	t.set_learning_rate(1.0f);
	t.w_props.size = w_props.size();
	t.w_props.ptr = w_props.data();
	t.d_props.size = d_props.size();
	t.d_props.ptr = d_props.data();	
	t.bias_props.size = bias_props.size();
	t.bias_props.ptr = bias_props.data();
	t.bias_deriv_props.size = bias_deriv.size();
	t.bias_deriv_props.ptr = bias_deriv.data();
	momentum_optimizer opt(0.0f);
	opt.init(&t);
	opt.update(&t);
	auto res = w_props.to_vector();
	std::vector<float> expected = { -9, -9 , -9, -9, -9 };
	compare_vectors(res, expected);
	res = bias_props.to_vector();
	expected = { -1, -1 , -1, -1, -1 };
	compare_vectors(res, expected);
	opt.update(&t);
	res = w_props.to_vector();
	expected = { -19, -19 , -19, -19, -19 };
	compare_vectors(res, expected);
	res = bias_props.to_vector();
	expected = { -3, -3 , -3, -3, -3 };
	compare_vectors(res, expected);
}

TEST(momentum_cuda_tests, clip_gradient_test1)
{
	device_vector<cuda_device, float> v;
	v.set_data({1, 2});
	float norm = cuda_vector_norm(v.data(), v.size());
	float expectected_norm = sqrt(5.0f);
	EXPECT_EQ(norm, expectected_norm);
	EXPECT_TRUE(cuda_scale_vector(v.data(), v.size(), 2.0f));
	auto v_res = v.to_vector();
	std::vector<float> expected_v = { 2.0f, 4.0f };
	compare_vectors(v_res, expected_v);
}

TEST(momentum_cuda_tests, test2)
{
	device_vector<cuda_device, float> w_props, d_props, bias_props, bias_deriv;
	w_props.set_data({ 1, 1, 1, 1, 1 });
	d_props.set_data({ 2, 2, 2, 2, 2 });
	bias_props.set_data({ 1, 1 });
	bias_deriv.set_data({ 2, 2 });
	test_layer t;
	t.set_learning_rate(1.0f);
	t.w_props.size = w_props.size();
	t.w_props.ptr = w_props.data();
	t.d_props.size = d_props.size();
	t.d_props.ptr = d_props.data();
	t.bias_props.size = bias_props.size();
	t.bias_props.ptr = bias_props.data();
	t.bias_deriv_props.size = bias_deriv.size();
	t.bias_deriv_props.ptr = bias_deriv.data();
	momentum_optimizer opt(0.0f, 1.0f);
	opt.init(&t);
	opt.update(&t);
	auto res = w_props.to_vector();
	float norm = sqrt(20.0f);
	float expected_val = 2 / norm;
	std::vector<float> expected = { 1 - expected_val, 1 - expected_val, 1 - expected_val, 1 - expected_val, 1 - expected_val };
	compare_vectors(res, expected);
	norm = sqrt(8.0f);
	expected_val = 2 / norm;
	res = bias_props.to_vector();
	expected = { 1 - expected_val, 1 - expected_val };
	compare_vectors(res, expected);
}

TEST(adam_cuda_tests, test1)
{
	cuda_floats m, v, w, d;
	std::vector<float> vals(4096 * 15, 0.0f);
	m.set_data(vals);
	v.set_data(vals);

	for (size_t i = 0; i < 1; i++)
	{
		std::generate(vals.begin(), vals.end(), [] { return 1.0f / rand(); });
		d.set_data(vals);
		std::generate(vals.begin(), vals.end(), [] { return 1.0f / rand(); });
		w.set_data(vals);
		
		auto m_temp = m.copy();
		auto v_temp = v.copy();
		auto w_temp = w.copy();

		adam_kernel(m.data(), v.data(), w.data(), d.data(), d.size(), 0.9, 0.999, 1 - 0.9, 1 - 0.999, 0.01);
		adam_kernel_vectorized(m_temp.data(), v_temp.data(), w_temp.data(), d.data(), d.size(), 0.9, 0.999, 1 - 0.9, 1 - 0.999, 0.01);
		compare_vectors(m.to_vector(), m_temp.to_vector());
		compare_vectors(v.to_vector(), v_temp.to_vector());
		compare_vectors(w.to_vector(), w_temp.to_vector());
	}
}