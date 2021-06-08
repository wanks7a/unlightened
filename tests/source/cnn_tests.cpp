#include <gtest/gtest.h>
#include <tests_objects.h>
#include <conv2d_cudnn.h>
#include <conv_transpose.h>

TEST(CNN_TESTS, cudnn_full_cnn_backprop_v1)
{
    test_layer  test;
    test.init_base(shape(3, 3, 1));
    test.set_output_shape(shape(3, 3, 1));
    test.output.setValues({
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
        });
    conv2d_cudnn cnn_l(3, 2);
    cnn_l.set_learning_rate(1.0f);
    auto opt = cnn_l.get_options();
    opt.zeropadding = true;
    cnn_l.set_options(opt);
    cnn_l.init_base(test.get_shape());

    EXPECT_TRUE(cnn_l.get_filters().get_weights().setValues({
        0, 0, 0,
        1, 1, 1,
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
        }));

    EXPECT_TRUE(cnn_l.get_bias_vector().setValues({
        0.0f, 0.0f
        }));

    cnn_l.forward_pass(&test);

    std::vector<float> expected = {
        1, 1, 0,
        1, 1, 1,
        0, 1, 1,
        2, 0, 0,
        0, 3, 0,
        0, 0, 2,
    };

    std::vector<float> result = cnn_l.get_native_output();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }

    test_layer  backprop_layer;
    backprop_layer.init_base(shape(3, 3, 2));
    backprop_layer.set_output_shape(shape(3, 3, 2));
    backprop_layer.output.setValues({
        0, 0, 0,
        1, 1, 1,
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
        });

    cnn_l.backprop(&backprop_layer);

    cnn_l.get_filters().get_weights_derivative().getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(CNN_TESTS, cudnn_full_cnn_backprop_v2)
{
    test_layer  test;
    test.init_base(shape(3, 3, 2));
    test.set_output_shape(shape(3, 3, 2));
    test.output.setValues({
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0, 1,
        0, 1, 0,
        1, 0, 0
        });
    conv2d_cudnn cnn_l(3, 2);
    cnn_l.set_learning_rate(1.0f);
    auto opt = cnn_l.get_options();
    opt.zeropadding = true;
    cnn_l.set_options(opt);
    cnn_l.init_base(test.get_shape());

    EXPECT_TRUE(cnn_l.get_filters().get_weights().setValues({
        0, 0, 0,
        1, 1, 1,
        0, 0, 0,
        0, 0, 0,
        1, 1, 1,
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
        }));

    EXPECT_TRUE(cnn_l.get_bias_vector().setValues({
        0, 0
        }));

    cnn_l.forward_pass(&test);

    std::vector<float> expected = {
        1, 2, 1,
        2, 2, 2,
        1, 2, 1,
        3, 0, 1,
        0, 4, 0,
        1, 0, 3,
    };

    std::vector<float> result = cnn_l.get_native_output();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }

    test_layer  backprop_layer;
    backprop_layer.init_base(shape(3, 3, 2));
    backprop_layer.set_output_shape(shape(3, 3, 2));
    backprop_layer.output.setValues({
        0, 0, 0,
        1, 1, 1,
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        });

    cnn_l.backprop(&backprop_layer);

    cnn_l.get_filters().get_weights_derivative().getCopy(result);

    expected = {
        1, 1, 0,
        1, 1, 1,
        0, 1, 1,
        0, 1, 1,
        1, 1, 1,
        1, 1, 0,
        2, 0, 0,
        0, 3, 0,
        0, 0, 2,
        1, 0, 1,
        0, 1, 0,
        1, 0, 1
    };

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(CNN_TESTS, cudnn_full_cnn_backprop_v2_batched)
{
    test_layer  test;
    test.init_base(shape(3, 3, 2, 2));
    test.set_output_shape(shape(3, 3, 2, 2));
    test.output.setValues({
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0, 1,
        0, 1, 0,
        1, 0, 0,
        0, 0, 1,
        0, 1, 0,
        1, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
        });
    conv2d_cudnn cnn_l(3, 2);
    auto opt = cnn_l.get_options();
    opt.zeropadding = true;
    cnn_l.set_options(opt);
    cnn_l.init_base(test.get_shape());
    cnn_l.set_learning_rate(1.0f);
    std::vector<float> weights_values = {
        0, 0, 0,
        1, 1, 1,
        0, 0, 0,
        0, 0, 0,
        1, 1, 1,
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };

    EXPECT_TRUE(cnn_l.get_filters().get_weights().setValues(weights_values));

    EXPECT_TRUE(cnn_l.get_bias_vector().setValues({
        0, 0
        }));

    cnn_l.forward_pass(&test);

    std::vector<float> expected = {
        1, 2, 1,
        2, 2, 2,
        1, 2, 1,
        3, 0, 1,
        0, 4, 0,
        1, 0, 3,
        1, 2, 1,
        2, 2, 2,
        1, 2, 1,
        3, 0, 1,
        0, 4, 0,
        1, 0, 3
    };

    std::vector<float> result = cnn_l.get_native_output();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }

    test_layer  backprop_layer;
    backprop_layer.init_base(shape(3, 3, 2, 2));
    backprop_layer.set_output_shape(shape(3, 3, 2, 2));
    backprop_layer.output.setValues({
        0, 0, 0,
        1, 1, 1,
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0, 0,
        1, 1, 1,
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        });

    cnn_l.backprop(&backprop_layer);

    cnn_l.get_filters().get_weights_derivative().getCopy(result);

    expected = {
        1, 2, 1,
        2, 2, 2,
        1, 2, 1,
        1, 2, 1,
        2, 2, 2,
        1, 2, 1, // first filter
        3, 0, 1,
        0, 4, 0,
        1, 0, 3,
        3, 0, 1,
        0, 4, 0,
        1, 0, 3, // second filter
    };

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(CNN_TESTS, cudnn_stride_v1)
{
    test_layer  test;
    test.init_base(shape(5, 5, 3));
    test.set_output_shape(shape(5, 5, 3));
    test.output.setValues({
        0, 0, 1, 0, 2,
        1, 0, 2, 0, 1,
        1, 0, 2, 2, 0,
        2, 0, 0, 2, 0,
        2, 1, 2, 2, 0, // end of 5x5
        2, 1, 2, 1, 1,
        2, 1, 2, 0, 1,
        0, 2, 1, 0, 1,
        1, 2, 2, 2, 2,
        0, 1, 2, 0, 1, // end of 5x5
        2, 1, 1, 2, 0,
        1, 0, 0, 1, 0,
        0, 1, 0, 0, 0,
        1, 0, 2, 1, 0,
        2, 2, 1, 1, 1,
        });
    filter_options opt;
    opt.zeropadding = true;
    opt.w = 3;
    opt.h = 3;
    opt.num_of_filters = 1;
    opt.channels = 3;
    opt.stride = 2;

    conv2d_cudnn cnn_l(opt);
    cnn_l.init_base(test.get_shape());
    cnn_l.set_learning_rate(1.0f);
    std::vector<float> weights_values = {
        -1, 0, 1,
        0, 0, 1,
        1, -1, 1,
        -1, 0, 1,
        1, -1, 1,
        0, 1, 0,
        -1, 1, 1,
        1, 1, 0,
        0, -1, 0,
    };

    EXPECT_TRUE(cnn_l.get_filters().get_weights().setValues(weights_values));

    EXPECT_TRUE(cnn_l.get_bias_vector().setValues({
        1
        }));

    cnn_l.forward_pass(&test);

    std::vector<float> expected = {
        2, 3, 3,
        3, 7, 3,
        8, 10, -3
    };

    std::vector<float> result = cnn_l.get_native_output();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(CNN_TESTS, cudnn_stride_v2)
{
    test_layer  test;
    test.init_base(shape(5, 5, 3));
    test.set_output_shape(shape(5, 5, 3));
    test.output.setValues({
        0, 0, 1, 0, 2,
        1, 0, 2, 0, 1,
        1, 0, 2, 2, 0,
        2, 0, 0, 2, 0,
        2, 1, 2, 2, 0, // end of 5x5
        2, 1, 2, 1, 1,
        2, 1, 2, 0, 1,
        0, 2, 1, 0, 1,
        1, 2, 2, 2, 2,
        0, 1, 2, 0, 1, // end of 5x5
        2, 1, 1, 2, 0,
        1, 0, 0, 1, 0,
        0, 1, 0, 0, 0,
        1, 0, 2, 1, 0,
        2, 2, 1, 1, 1,
        });
    filter_options opt;
    opt.zeropadding = true;
    opt.w = 3;
    opt.h = 3;
    opt.num_of_filters = 2;
    opt.channels = 3;
    opt.stride = 2;

    conv2d_cudnn cnn_l(opt);
    cnn_l.init_base(test.get_shape());
    cnn_l.set_learning_rate(1.0f);
    std::vector<float> weights_values = {
        -1, 0, 1,
        0, 0, 1,
        1, -1, 1,
        -1, 0, 1,
        1, -1, 1,
        0, 1, 0,
        -1, 1, 1,
        1, 1, 0,
        0, -1, 0,
        0, 1, -1,
        0, -1, 0,
        0, -1, 1,
        -1, 0, 0,
        1, -1, 0,
        1, -1, 0,
        -1, 1, -1,
        0, -1, -1,
        1, 0, 0
    };

    EXPECT_TRUE(cnn_l.get_filters().get_weights().setValues(weights_values));

    EXPECT_TRUE(cnn_l.get_bias_vector().setValues({
        1, 0
        }));

    cnn_l.forward_pass(&test);

    std::vector<float> expected = {
        2, 3, 3,
        3, 7, 3,
        8, 10, -3,
        -8, -8, -3,
        -3, 1, 0,
        -3, -8, -5
    };

    std::vector<float> result = cnn_l.get_native_output();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}