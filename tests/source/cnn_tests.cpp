#include <gtest/gtest.h>
#include <cnn_layer.h>
#include <tests_objects.h>

TEST(CNN_TESTS, full_cnn_backprop_v1)
{
    test_layer  test;
    test.init_base(shape(3, 3, 1));
    test.set_output_shape(shape(3, 3, 1));
    test.output.setValues({
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
        });
    cnn_layer cnn_l(3, 2);
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
        0, 0
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

TEST(CNN_TESTS, full_cnn_backprop_v2)
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
    cnn_layer cnn_l(3, 2);
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

TEST(CNN_TESTS, full_cnn_backprop_v2_batched)
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
    cnn_layer cnn_l(3, 2);
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
        1, 1, 0,
        1, 1, 1,
        0, 1, 1,
        0, 1, 1,
        1, 1, 1,
        1, 1, 0,
        0, 1, 1,
        1, 1, 1,
        1, 1, 0,
        1, 1, 0,
        1, 1, 1,
        0, 1, 1, // every batch for the first weights only
        2, 0, 0,
        0, 3, 0,
        0, 0, 2,
        1, 0, 1,
        0, 1, 0,
        1, 0, 1,
        1, 0, 1,
        0, 1, 0,
        1, 0, 1,
        2, 0, 0,
        0, 3, 0,
        0, 0, 2, // and now 2 batches for the 2nd weights
    };

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}