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

    std::vector<float> expected_weights = {
        -1, -1, 0,
        0, 0, 0,
        0, -1, -1,
        -1, 0, 0,
        0, -2, 0,
        0,  0, -1
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

    cnn_l.get_filters().get_weights().getCopy(result);
    EXPECT_EQ(result.size(), expected_weights.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected_weights[i]);
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


    // testing the weights update here
    expected = {
        -1, -1, 0,
        0, 0, 0,
        0, -1, -1,
        0, -1, -1,
        0, 0, 0,
        -1, -1, 0,
        -1, 0, 0,
        0, -2, 0,
        0, 0, -1,
        0, 0, -1,
        0, 0, 0,
        -1, 0, 0
    };

    cnn_l.get_filters().get_weights().getCopy(result);

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
        0, 1, 1, // every batch for the first filter weights only
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
        0, 0, 2, // and now the batches for the 2nd filter
    };

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }

    // testing the weights update here
    expected = {
        1, 2, 1,
        2, 2, 2,
        1, 2, 1,
        1, 2, 1,
        2, 2, 2,
        1, 2, 1,
        3, 0, 1,
        0, 4, 0,
        1, 0, 3,
        3, 0, 1,
        0, 4, 0,
        1, 0, 3
    };

    for (size_t i = 0; i < expected.size(); i++)
    {
        expected[i] = weights_values[i] - expected[i] / 2.0f;
    }

    cnn_l.get_filters().get_weights().getCopy(result);

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(CNN_TESTS, backprop_input_v1)
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
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0, 0,
        1, 1, 1,
        0, 0, 0,
        0, 0, 1,
        0, 1, 0,
        1, 0, 0
        }));
    cnn_l.forward_pass(&test);

    test_layer  backprop_l;
    backprop_l.init_base(shape(3, 3, 2));
    backprop_l.set_output_shape(shape(3, 3, 2));
    backprop_l.output.setValues({
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0, 1,
        0, 1, 0,
        1, 0, 0
        });
    cnn_l.backprop(&backprop_l);

    std::vector<float> result = cnn_l.get_native_derivative();
    
    std::vector<float> expected = {
        1, 2, 1,
        2, 2, 2,
        1, 2, 1,
        2, 0, 2,
        0, 6, 0,
        2, 0, 2,
    };

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(CNN_TESTS, backprop_input_v2)
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
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0, 1,
        0, 1, 0,
        1, 0, 0,
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
        0, 0, 1,
        0, 0, 0,
        1, 1, 1,
        0, 0, 0,
        0, 0, 1,
        0, 1, 0,
        1, 0, 0
        }));
    cnn_l.forward_pass(&test);

    test_layer  backprop_l;
    backprop_l.init_base(shape(3, 3, 2, 2));
    backprop_l.set_output_shape(shape(3, 3, 2, 2));
    backprop_l.output.setValues({
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0, 1,
        0, 1, 0,
        1, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0, 1,
        0, 1, 0,
        1, 0, 0
        });
    cnn_l.backprop(&backprop_l);

    std::vector<float> result = cnn_l.get_native_derivative();

    std::vector<float> expected = {
        1, 2, 1,
        2, 2, 2,
        1, 2, 1,
        2, 0, 2,
        0, 6, 0,
        2, 0, 2,
        1, 2, 1,
        2, 2, 2,
        1, 2, 1,
        2, 0, 2,
        0, 6, 0,
        2, 0, 2
    };

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(CNN_TESTS, flip_test_horizontal_v1)
{
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1, 0, 0,
        0, 1, 0,
        2, 0, 1
        }));
    shape input_shape;
    input_shape.width = 3;
    input_shape.height = 3;
    flip_filter(input.get(), input_shape, true);

    std::vector<float> expected = {
        2,0,1,
        0,1,0,
        1,0,0
    };
    std::vector<float> result;
    input.getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(CNN_TESTS, flip_test_horizontal_v2)
{
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1, 0, 0,
        0, 1, 0,
        2, 0, 1,
        1, 0, 0,
        0, 1, 0,
        2, 0, 1
        }));
    shape input_shape;
    input_shape.width = 3;
    input_shape.height = 3;
    input_shape.depth = 2;
    flip_filter(input.get(), input_shape, true);

    std::vector<float> expected = {
        2,0,1,
        0,1,0,
        1,0,0,
        2,0,1,
        0,1,0,
        1,0,0
    };
    std::vector<float> result;
    input.getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(CNN_TESTS, flip_test_horizontal_v3)
{
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1, 0, 0,
        0, 1, 0,
        2, 0, 1,
        1, 0, 0,
        0, 1, 0,
        2, 0, 1,
        1, 0, 0,
        0, 1, 0,
        2, 0, 1,
        1, 0, 0,
        0, 1, 0,
        2, 0, 1,
        }));
    shape input_shape;
    input_shape.width = 3;
    input_shape.height = 3;
    input_shape.depth = 2;
    input_shape.batches = 2;

    flip_filter(input.get(), input_shape, true);

    std::vector<float> expected = {
        2,0,1,
        0,1,0,
        1,0,0,
        2,0,1,
        0,1,0,
        1,0,0,
        2,0,1,
        0,1,0,
        1,0,0,
        2,0,1,
        0,1,0,
        1,0,0
    };
    std::vector<float> result;
    input.getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(CNN_TESTS, flip_test_vertical_v1)
{
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1, 0, 0,
        0, 1, 0,
        2, 0, 1
        }));
    shape input_shape;
    input_shape.width = 3;
    input_shape.height = 3;
    flip_filter(input.get(), input_shape, false);

    std::vector<float> expected = {
        0,0,1,
        0,1,0,
        1,0,2
    };
    std::vector<float> result;
    input.getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(CNN_TESTS, flip_test_vertical_v2)
{
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1, 0, 0,
        0, 1, 0,
        2, 0, 1,
        1, 0, 0,
        0, 1, 0,
        2, 0, 1,
        }));
    shape input_shape;
    input_shape.width = 3;
    input_shape.height = 3;
    input_shape.depth = 2;
    flip_filter(input.get(), input_shape, false);

    std::vector<float> expected = {
        0,0,1,
        0,1,0,
        1,0,2,
        0,0,1,
        0,1,0,
        1,0,2,
    };
    std::vector<float> result;
    input.getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(CNN_TESTS, flip_test_vertical_v3)
{
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1, 0, 0,
        0, 1, 0,
        2, 0, 1,
        1, 0, 0,
        0, 1, 0,
        2, 0, 1,
        1, 0, 0,
        0, 1, 0,
        2, 0, 1,
        1, 0, 0,
        0, 1, 0,
        2, 0, 1,
        }));
    shape input_shape;
    input_shape.width = 3;
    input_shape.height = 3;
    input_shape.depth = 2;
    input_shape.batches = 2;
    flip_filter(input.get(), input_shape, false);

    std::vector<float> expected = {
        0,0,1,
        0,1,0,
        1,0,2,
        0,0,1,
        0,1,0,
        1,0,2,
        0,0,1,
        0,1,0,
        1,0,2,
        0,0,1,
        0,1,0,
        1,0,2,
    };
    std::vector<float> result;
    input.getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(CNN_TESTS, bias_update_v1)
{
    shape deriv_shape(3, 3, 2, 1);
    cuVector<float> derivative;
    EXPECT_TRUE(derivative.setValues(
        {
            1,2,3,
            4,5,6,
            7,8,9,
            1,1,1,
            2,2,2,
            3,3,3
        }));
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({
        0, 0
        }));
    update_bias(derivative.get(), deriv_shape, bias.get(), 1.0f);

    std::vector<float> expected = {
        -5, -2
    };

    std::vector<float> result;
    bias.getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(CNN_TESTS, bias_update_v2)
{
    shape deriv_shape(3, 3, 2, 2);
    cuVector<float> derivative;
    EXPECT_TRUE(derivative.setValues(
        {
            1,2,3,
            4,5,6,
            7,8,9,
            1,1,1,
            2,2,2,
            3,3,3,
            1,2,3,
            4,5,6,
            7,8,9,
            1,1,1,
            2,2,2,
            3,3,3,
        }));
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({
        0, 0
        }));
    update_bias(derivative.get(), deriv_shape, bias.get(), 1.0f);

    std::vector<float> expected = {
        -5, -2
    };

    std::vector<float> result;
    bias.getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(CNN_TESTS, bias_update_v3)
{
    shape deriv_shape(3, 3, 1);
    cuVector<float> derivative;
    EXPECT_TRUE(derivative.setValues(
        {
            1,2,3,
            4,5,6,
            7,8,9
        }));
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({
        0
        }));
    update_bias(derivative.get(), deriv_shape, bias.get(), 1.0f);

    std::vector<float> expected = {
        -5
    };

    std::vector<float> result;
    bias.getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}