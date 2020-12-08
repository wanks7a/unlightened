#include <gtest/gtest.h>
#include <tests_objects.h>
#include <conv_transpose.h>

TEST(transposed_conv_tests, test1)
{
	conv2d_transposed l(1, 3, 1, conv2d_transposed::padding::VALID);
	shape input(1, 1);
	l.init_base(input);
    EXPECT_TRUE(l.get_filter().get_weights().setValues({
        1,2,3,
        4,5,6,
        7,8,9 }));
    test_layer  test;
    test.init_base(input);
    test.set_output_shape(input);
    test.output.setValues({
        1
        });
    l.forward_pass(&test);
    auto res = l.get_native_output();
    std::vector<float> expected = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    compare_vectors(res, expected);
    test.init_base(shape(3, 3));
    test.set_output_shape(shape(3, 3));
    test.output.setValues({
        1,1,1,
        1,1,1,
        1,1,1
        });
    l.backprop(&test);
    res = l.get_native_derivative();
    expected = {
        45
    };
    compare_vectors(res, expected);
}

TEST(transposed_conv_tests, test2)
{
    conv2d_transposed l(1, 3, 1, conv2d_transposed::padding::VALID);
    shape input(1, 1, 1, 2);
    l.init_base(input);
    l.get_filter().get_weights().setValues({
        1,2,3,
        4,5,6,
        7,8,9 });
    test_layer  test;
    test.init_base(input);
    test.set_output_shape(input);
    test.output.setValues({
        1,1
        });
    l.forward_pass(&test);
    auto res = l.get_native_output();
    std::vector<float> expected = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    compare_vectors(res, expected);
    test.init_base(shape(3, 3, 1, 2));
    test.set_output_shape(shape(3, 3, 1, 2));
    test.output.setValues({
        1,1,1,
        1,1,1,
        1,1,1,
        1,1,1,
        1,1,1,
        1,1,1
        });
    l.backprop(&test);
    res = l.get_native_derivative();
    expected = {
        45, 45
    };
    compare_vectors(res, expected);
}