#include <conv_filter.h>
#include <GpuUtils.h>
#include <device_memory.h>
#include <LinearLayer.h>
#include <NeuralNet.h>
#include <SigmoidLayerGPU.h>
#include <LinearLayerGPU.h>
#include <array>
#include <unordered_map>
#include <gtest/gtest.h>
#include <max_pool.h>
#include <activation_layer.h>
#include <cmath>
#include <csv_parser.h>
#include <tests_objects.h>

class gpu_tests : public ::testing::Test 
{
protected:
    //virtual void SetUp() 
    //{
    //    ASSERT_TRUE(utils::GpuInit());
    //}

    //virtual void TearDown() 
    //{
    //    ASSERT_TRUE(utils::GpuRelase());
    //}
};

TEST(gpu_tests, activation_2d)
{
    test_layer  test;
    test.init_base(shape(3, 3));
    test.set_output_shape(shape(3, 3));
    EXPECT_TRUE(test.output.setValues({
        1,1,1,
        2,2,2,
        3,3,3
        }));
    activation_layer act(activation_layer::activation_function::Identity);
    act.init_base(test.get_shape());
    act.forward_pass(&test);
    std::vector<float> expected = {
        1,1,1,
        2,2,2,
        3,3,3
    };

    std::vector<float> result = act.get_native_output();
   
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
    EXPECT_TRUE(test.output.setValues({
        5,5,5,
        6,6,6,
        7,7,7
        }));
    expected = {
        5,5,5,
        6,6,6,
        7,7,7
    };
    act.backprop(&test);
    result = act.get_native_derivative();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}


TEST(gpu_tests, activation_3d)
{
    test_layer  test;
    test.init_base(shape(3, 3, 2));
    test.set_output_shape(shape(3, 3, 2));
    test.output.setValues({
        1,1,1,
        2,2,2,
        3,3,3,
        5,5,5,
        6,6,6,
        7,7,7
        });
    activation_layer act(activation_layer::activation_function::Identity);
    act.init_base(test.get_shape());
    act.forward_pass(&test);
    std::vector<float> expected = {
        1,1,1,
        2,2,2,
        3,3,3,
        5,5,5,
        6,6,6,
        7,7,7
    };

    std::vector<float> result = act.get_native_output();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }

    EXPECT_TRUE(test.output.setValues({
        5,5,5,
        6,6,6,
        7,7,7,
        8,8,8,
        9,9,9,
        1,1,1
        }));
    expected = {
        5,5,5,
        6,6,6,
        7,7,7,
        8,8,8,
        9,9,9,
        1,1,1
    };
    act.backprop(&test);
    result = act.get_native_derivative();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, activation_3d_batch2)
{
    test_layer  test;
    test.init_base(shape(3, 3, 2, 2));
    test.set_output_shape(shape(3, 3, 2, 2));
    test.output.setValues({
        1,1,1,
        2,2,2,
        3,3,3,
        5,5,5,
        6,6,6,
        7,7,7,
        10,10,10,
        11,11,11,
        12,12,12,
        13,13,13,
        14,14,14,
        15,15,15
        });
    activation_layer act(activation_layer::activation_function::Identity);
    act.init_base(test.get_shape());
    act.forward_pass(&test);
    std::vector<float> expected = {
        1,1,1,
        2,2,2,
        3,3,3,
        5,5,5,
        6,6,6,
        7,7,7,
        10,10,10,
        11,11,11,
        12,12,12,
        13,13,13,
        14,14,14,
        15,15,15
    };

    std::vector<float> result = act.get_native_output();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }

    EXPECT_TRUE(test.output.setValues({
        5,5,5,
        6,6,6,
        7,7,7,
        8,8,8,
        9,9,9,
        1,1,1,
        5,5,5,
        6,6,6,
        7,7,7,
        8,8,8,
        9,9,9,
        1,1,1
        }));
    expected = {
        5,5,5,
        6,6,6,
        7,7,7,
        8,8,8,
        9,9,9,
        1,1,1,
        5,5,5,
        6,6,6,
        7,7,7,
        8,8,8,
        9,9,9,
        1,1,1
    };
    act.backprop(&test);
    result = act.get_native_derivative();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, max_pool_back_prop_depth1)
{
    shape input_shape(3, 3);
    shape output_shape(5, 5);

    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,2,3,
        4,5,6,
        7,8,9
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size(), 20.0f));
    cuVector<char> mask;
    mask.setValues({
        0, 1, 2,
        3, 0, 2,
        1, 0, 0
        });

    std::vector<float> expected = {
        1, 0, 0, 2, 0,
        0, 0, 0, 0, 3,
        0, 0, 5, 0, 0,
        0, 4, 0, 0, 6,
        0, 7, 8, 0, 9
    };

    max_pooling_backprop(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
    std::vector<float> result;
    output.getCopy(result);
    EXPECT_EQ(expected.size(), result.size());
    for (size_t i = 0; i < expected.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
    }
}

TEST(gpu_tests, max_pool_back_prop_depth2)
{
    shape input_shape(3, 3, 2);
    shape output_shape(5, 5, 2);

    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,2,3,
        4,5,6,
        7,8,9,
        9,8,7,
        6,5,4,
        3,2,1,
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size(), 20.0f));
    cuVector<char> mask;
    mask.setValues({
        0, 1, 2,
        3, 0, 2,
        1, 0, 0,
        1, 3, 2,
        2, 0, 0,
        1, 1, 0
        });

    std::vector<float> expected = {
        1, 0, 0, 2, 0,
        0, 0, 0, 0, 3,
        0, 0, 5, 0, 0,
        0, 4, 0, 0, 6,
        0, 7, 8, 0, 9,
        0, 9, 0, 0, 0,
        0, 0, 0, 8, 7,
        0, 0, 5, 0, 4,
        6, 0, 0, 0, 0,
        0, 3, 0, 2, 1
    };

    max_pooling_backprop(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
    std::vector<float> result;
    output.getCopy(result);
    EXPECT_EQ(expected.size(), result.size());
    for (size_t i = 0; i < expected.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
    }
}

TEST(gpu_tests, max_pool_back_prop_depth2_batch2)
{
    shape input_shape(3, 3, 2, 2);
    shape output_shape(5, 5, 2, 2);

    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,2,3,
        4,5,6,
        7,8,9,
        9,8,7,
        6,5,4,
        3,2,1,
        9,8,7,
        6,5,4,
        3,2,1,
        1,2,3,
        4,5,6,
        7,8,9
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size(), 20.0f)); 
    cuVector<char> mask;
    mask.setValues({
        0, 1, 2,
        3, 0, 2,
        1, 0, 0,
        1, 3, 2,
        2, 0, 0,
        1, 1, 0,
        1, 3, 2,
        2, 0, 0,
        1, 1, 0,
        0, 1, 2,
        3, 0, 2,
        1, 0, 0
        });

    std::vector<float> expected = {
        1, 0, 0, 2, 0,
        0, 0, 0, 0, 3,
        0, 0, 5, 0, 0,
        0, 4, 0, 0, 6,
        0, 7, 8, 0, 9,
        0, 9, 0, 0, 0,
        0, 0, 0, 8, 7,
        0, 0, 5, 0, 4,
        6, 0, 0, 0, 0,
        0, 3, 0, 2, 1,
        0, 9, 0, 0, 0,
        0, 0, 0, 8, 7,
        0, 0, 5, 0, 4,
        6, 0, 0, 0, 0,
        0, 3, 0, 2, 1,
        1, 0, 0, 2, 0,
        0, 0, 0, 0, 3,
        0, 0, 5, 0, 0,
        0, 4, 0, 0, 6,
        0, 7, 8, 0, 9
    };

    max_pooling_backprop(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
    std::vector<float> result;
    output.getCopy(result);
    EXPECT_EQ(expected.size(), result.size());
    for (size_t i = 0; i < expected.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
    }
}

TEST(gpu_tests, max_pool_depth1)
{
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,0,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,1,0,0,0,
        0,0,0,0,0
        }));
    shape input_shape;
    input_shape.width = 5;
    input_shape.height = 5;
    input_shape.depth = 1;
    shape output_shape;
    output_shape.width = 3;
    output_shape.height = 3;
    output_shape.depth = 1;
    cuVector<char> mask;
    EXPECT_TRUE(mask.resize(output_shape.size()));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size(), 0.0f));

    std::vector<float> expected =
    {
        1, 1, 0,
        1, 1, 0,
        0, 0, 0
    };
    std::vector<int> mask_expected =
    {
        0, 2, 0,
        3, 1, 0,
        0, 0, 0
    };
    max_pooling(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<char> mask_result;
    mask.getCopy(mask_result);
    EXPECT_EQ(expected.size(), result.size());
    EXPECT_EQ(mask_expected.size(), mask_result.size());
    for (size_t i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
        EXPECT_EQ(mask_expected[i], mask_result[i]);
    }
}

TEST(gpu_tests, max_pool_depth2)
{
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,0,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,1,0,0,0,
        0,0,0,0,0,
        0,1,0,0,0,
        0,0,0,1,0,
        0,0,0,0,0,
        0,1,1,0,0,
        0,0,0,0,0
    }));
    shape input_shape;
    input_shape.width = 5;
    input_shape.height = 5;
    input_shape.depth = 2;
    shape output_shape;
    output_shape.width = 3;
    output_shape.height = 3;
    output_shape.depth = 2;
    cuVector<char> mask;
    EXPECT_TRUE(mask.resize(output_shape.size()));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size(), 0.0f));

    std::vector<float> expected =
    {
        1, 1, 0,
        1, 1, 0,
        0, 0, 0,
        1, 1, 0,
        1, 1, 0,
        0, 0, 0
    };
    std::vector<int> mask_expected =
    {
        0, 2, 0,
        3, 1, 0,
        0, 0, 0,
        1, 3, 0,
        3, 2, 0,
        0, 0, 0
    };
    max_pooling(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<char> mask_result;
    mask.getCopy(mask_result);
    EXPECT_EQ(expected.size(), result.size());
    EXPECT_EQ(mask_expected.size(), mask_result.size());
    for (size_t i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
        EXPECT_EQ(mask_expected[i], mask_result[i]);
    }
}

TEST(gpu_tests, max_pool_depth2_batch2)
{
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,0,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,1,0,0,0,
        0,0,0,0,0,
        0,1,0,0,0,
        0,0,0,1,0,
        0,0,0,0,0,
        0,1,1,0,0,
        0,0,0,0,0,
        1,0,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,1,0,0,0,
        0,0,0,0,0,
        0,1,0,0,0,
        0,0,0,1,0,
        0,0,0,0,0,
        0,1,1,0,0,
        0,0,0,0,0,
        }));
    shape input_shape;
    input_shape.width = 5;
    input_shape.height = 5;
    input_shape.depth = 2;
    input_shape.batches = 2;
    shape output_shape;
    output_shape.width = 3;
    output_shape.height = 3;
    output_shape.depth = 2;
    output_shape.batches = 2;
    cuVector<char> mask;
    EXPECT_TRUE(mask.resize(output_shape.size()));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size(), 0.0f));

    std::vector<float> expected =
    {
        1, 1, 0,
        1, 1, 0,
        0, 0, 0,
        1, 1, 0,
        1, 1, 0,
        0, 0, 0,
        1, 1, 0,
        1, 1, 0,
        0, 0, 0,
        1, 1, 0,
        1, 1, 0,
        0, 0, 0,
    };
    std::vector<int> mask_expected =
    {
        0, 2, 0,
        3, 1, 0,
        0, 0, 0,
        1, 3, 0,
        3, 2, 0,
        0, 0, 0,
        0, 2, 0,
        3, 1, 0,
        0, 0, 0,
        1, 3, 0,
        3, 2, 0,
        0, 0, 0,
    };
    max_pooling(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<char> mask_result;
    mask.getCopy(mask_result);
    EXPECT_EQ(expected.size(), result.size());
    EXPECT_EQ(mask_expected.size(), mask_result.size());
    for (size_t i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
        EXPECT_EQ(mask_expected[i], mask_result[i]);
    }
}

int main(int argc, char** argv)
{
    EXPECT_TRUE(utils::GpuInit());
    ::testing::InitGoogleTest(&argc, argv);
    EXPECT_TRUE(utils::GpuRelase());
    return RUN_ALL_TESTS();
}