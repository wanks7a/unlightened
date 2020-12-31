#include <tests_objects.h>
#include <batch_norm_cuda.h>

TEST(cuda_batch_norm_test_1, test1)
{
	batch_norm_cuda b;
	shape input_shape(5, 5, 3, 5);
	b.init_base(input_shape);
	test_layer t;
	t.set_output_shape(input_shape);
	std::vector<float> input_data;
	input_data.resize(input_shape.size(), 5.0f);
	for (size_t i = 0; i < input_data.size(); i++)
	{
		input_data[i] = i;
	}
	t.output.setValues(input_data);
	b.forward_pass(&t);
	auto result = b.get_native_output();
	b.backprop(&t);
	result = b.get_native_derivative();
}