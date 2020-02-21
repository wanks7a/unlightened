#include <conv_filter.h>
#include <GpuUtils.h>
#include <GpuMemory.h>
#include <LinearLayer.h>
#include <NeuralNet.h>
#include <SigmoidLayerGPU.h>
#include <cnn_layer.h>
#include <LinearLayerGPU.h>
#include <array>
#include <unordered_map>
#include <gtest/gtest.h>
#include <max_pool.h>

class gpu_tests : public ::testing::Test 
{
protected:
    virtual void SetUp() 
    {
        ASSERT_TRUE(utils::GpuInit());
    }

    virtual void TearDown() 
    {
        ASSERT_TRUE(utils::GpuRelase());
    }
};

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

    max_pool_backprop(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
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

    max_pool_backprop(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
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

    max_pool_backprop(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
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
    max_pool(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
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
    max_pool(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
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
    max_pool(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
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

TEST(gpu_tests, cnn_conv_3d_depth1_same)
{
    shape input_shape;
    input_shape.width = 5;
    input_shape.height = 5;
    shape output_shape;
    output_shape.width = 5;
    output_shape.height = 5;
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2
    }));
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.area()));
    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), 3, true);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<float> expected = { 0, 1, 2, 2, 2,
                                  -4, -4, -4, -4, -2,
                                    0, -1, -2, -1, 0,
                                    -1, 1, 2, 1, 2,
                                     -4, -4, -4, -5, -2 };
    EXPECT_EQ(expected.size(), result.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
    }
}

TEST(gpu_tests, cnn_conv_3d_depth3_same)
{
    shape input_shape;
    input_shape.width = 5;
    input_shape.height = 5;
    input_shape.depth = 3;
    shape output_shape;
    output_shape.width = 5;
    output_shape.height = 5;
    output_shape.depth = 3;
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2
        }));
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1,
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1,
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.volume()));
    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), 3, true);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<float> expected = { 
                        0, 1, 2, 2, 2,
                        -4, -4, -4, -4, -2,
                        0, -1, -2, -1, 0,
                        -1, 1, 2, 1, 2,
                        -4, -4, -4, -5, -2,
                        0, 1, 2, 2, 2,
                        -4, -4, -4, -4, -2,
                        0, -1, -2, -1, 0,
                        -1, 1, 2, 1, 2,
                        -4, -4, -4, -5, -2, 
                        0, 1, 2, 2, 2,
                        -4, -4, -4, -4, -2,
                        0, -1, -2, -1, 0,
                        -1, 1, 2, 1, 2,
                        -4, -4, -4, -5, -2,
    };
    EXPECT_EQ(expected.size(), result.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
    }
}

TEST(gpu_tests, cnn_conv_3d_depth3_batch2_same)
{
    shape input_shape;
    input_shape.width = 5;
    input_shape.height = 5;
    input_shape.depth = 3;
    input_shape.batches = 2;
    shape output_shape;
    output_shape.width = 5;
    output_shape.height = 5;
    output_shape.depth = 3;
    output_shape.batches = 2;
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
        }));
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1,
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1,
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size()));
    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), 3, true);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<float> expected = {
                        0, 1, 2, 2, 2,
                        -4, -4, -4, -4, -2,
                        0, -1, -2, -1, 0,
                        -1, 1, 2, 1, 2,
                        -4, -4, -4, -5, -2,
                        0, 1, 2, 2, 2,
                        -4, -4, -4, -4, -2,
                        0, -1, -2, -1, 0,
                        -1, 1, 2, 1, 2,
                        -4, -4, -4, -5, -2,
                        0, 1, 2, 2, 2,
                        -4, -4, -4, -4, -2,
                        0, -1, -2, -1, 0,
                        -1, 1, 2, 1, 2,
                        -4, -4, -4, -5, -2,
                        0, 1, 2, 2, 2,
                        -4, -4, -4, -4, -2,
                        0, -1, -2, -1, 0,
                        -1, 1, 2, 1, 2,
                        -4, -4, -4, -5, -2,
                        0, 1, 2, 2, 2,
                        -4, -4, -4, -4, -2,
                        0, -1, -2, -1, 0,
                        -1, 1, 2, 1, 2,
                        -4, -4, -4, -5, -2,
                        0, 1, 2, 2, 2,
                        -4, -4, -4, -4, -2,
                        0, -1, -2, -1, 0,
                        -1, 1, 2, 1, 2,
                        -4, -4, -4, -5, -2
    };
    EXPECT_EQ(expected.size(), result.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
    }
}

TEST(gpu_tests, cnn_conv_3d_depth1_valid)
{
    shape input_shape;
    input_shape.width = 6;
    input_shape.height = 6;
    shape output_shape;
    output_shape.width = 4;
    output_shape.height = 4;
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
    }));
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({
                        1,1,1,
                        0,0,0,
                        -1,-1,-1
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(static_cast<size_t>(output_shape.width) * output_shape.height));
    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), 3, false);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<float> expected = { 0,0,0,0,
                                    30,10,-10,-30,
                                    30,10,-10,-30,
                                    0,0,0,0
    };
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, cnn_conv_3d_depth3_valid)
{
    shape input_shape;
    input_shape.width = 6;
    input_shape.height = 6;
    input_shape.depth = 3;
    shape output_shape;
    output_shape.width = 4;
    output_shape.height = 4;
    output_shape.depth = 3;
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
        }));
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({
                        1,1,1,
                        0,0,0,
                        -1,-1,-1,
                        1,1,1,
                        0,0,0,
                        -1,-1,-1,
                        1,1,1,
                        0,0,0,
                        -1,-1,-1
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.volume()));
    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), 3, false);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<float> expected = { 0,0,0,0,
                                    30,10,-10,-30,
                                    30,10,-10,-30,
                                    0,0,0,0,
                                    0,0,0,0,
                                    30,10,-10,-30,
                                    30,10,-10,-30,
                                    0,0,0,0,
                                    0,0,0,0,
                                    30,10,-10,-30,
                                    30,10,-10,-30,
                                    0,0,0,0,
    };
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
    }
}

TEST(gpu_tests, cnn_conv_3d_depth3_batch2_valid)
{
    shape input_shape;
    input_shape.width = 6;
    input_shape.height = 6;
    input_shape.depth = 3;
    input_shape.batches = 2;
    shape output_shape;
    output_shape.width = 4;
    output_shape.height = 4;
    output_shape.depth = 3;
    output_shape.batches = 2;
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
        }));
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({
                        1,1,1,
                        0,0,0,
                        -1,-1,-1,
                        1,1,1,
                        0,0,0,
                        -1,-1,-1,
                        1,1,1,
                        0,0,0,
                        -1,-1,-1
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size()));
    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), 3, false);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<float> expected = { 0,0,0,0,
                                    30,10,-10,-30,
                                    30,10,-10,-30,
                                    0,0,0,0,
                                    0,0,0,0,
                                    30,10,-10,-30,
                                    30,10,-10,-30,
                                    0,0,0,0,
                                    0,0,0,0,
                                    30,10,-10,-30,
                                    30,10,-10,-30,
                                    0,0,0,0,
                                    0,0,0,0,
                                    30,10,-10,-30,
                                    30,10,-10,-30,
                                    0,0,0,0,
                                    0,0,0,0,
                                    30,10,-10,-30,
                                    30,10,-10,-30,
                                    0,0,0,0,
                                    0,0,0,0,
                                    30,10,-10,-30,
                                    30,10,-10,-30,
                                    0,0,0,0,
    };
    EXPECT_EQ(expected.size(), result.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
    }
}

TEST(gpu_tests, cnn_full_conv_2d_001)
{
    shape input_shape(5, 5);
    shape output_shape(6, 6);
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({ 1, 1, 1, 1 }));
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5
        }));

    cuVector<float> output;
    EXPECT_TRUE(output.resize(6 * 6));
    std::vector<float> expected = {
        1,3,5,7,9,5,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        1,3,5,7,9,5
    };
    full_conv_2d(input.get(), input_shape, output.get(), output_shape, weights.get(), 2);
    std::vector<float> result;
    output.getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, cnn_full_conv_2d_002)
{
    shape input_shape(5, 5, 3);
    shape output_shape(6, 6, 3);
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({ 1, 1, 1, 1 }));
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5
        }));

    cuVector<float> output;
    EXPECT_TRUE(output.resize(6 * 6 * 3));
    std::vector<float> expected = {
        1,3,5,7,9,5,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        1,3,5,7,9,5,
        1,3,5,7,9,5,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        1,3,5,7,9,5,
        1,3,5,7,9,5,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        1,3,5,7,9,5
    };
    full_conv_2d(input.get(), input_shape, output.get(), output_shape, weights.get(), 2);
    std::vector<float> result;
    output.getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, cnn_full_conv_2d_003)
{
    shape input_shape(5, 5, 3, 2);
    shape output_shape(6, 6, 3, 2);
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({ 1, 1, 1, 1 }));
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        }));

    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size()));
    std::vector<float> expected = {
        1,3,5,7,9,5,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        1,3,5,7,9,5,
        1,3,5,7,9,5,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        1,3,5,7,9,5,
        1,3,5,7,9,5,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        1,3,5,7,9,5,
        1,3,5,7,9,5,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        1,3,5,7,9,5,
        1,3,5,7,9,5,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        1,3,5,7,9,5,
        1,3,5,7,9,5,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        1,3,5,7,9,5,
    };
    full_conv_2d(input.get(), input_shape, output.get(), output_shape, weights.get(), 2);
    std::vector<float> result;
    output.getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, cnn_basic)
{
    shape input_shape;
    input_shape.width = 7;
    input_shape.height = 7;
    cnn_layer layer(3, 5);
    auto options = layer.get_options();
    options.zeropadding = false;
    layer.set_options(options);
    layer.init(input_shape);
    EXPECT_EQ(layer.get_shape().depth, 5);
    EXPECT_EQ(layer.get_shape().height, 5);
    EXPECT_EQ(layer.get_shape().width, 5);
}

TEST(gpu_tests, dense_layer1024)
{
    std::vector<float> input;
    input.resize(1024 * 1024, 1.0f);
    LinearLayerGPU<false> l(1024);
    shape input_shape(1, 1024);
    l.init(input_shape);
    EXPECT_EQ(l.get_shape().size(), 1024 + 1); // because of bias
    EXPECT_TRUE(l.set_weights(input));
    cuVector<float> inputPtr;
    inputPtr.setValues(input);
    cuVector<float> weightsGPU;
    weightsGPU.setValues(input);
    cuVector<float> outputGPU;
    EXPECT_TRUE(outputGPU.resize(1025));
    std::vector<float> result;
    for (size_t i = 0; i < 1000; i++)
    {
        linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr.get(), 1024, 1024, true);
    }
    outputGPU.getCopy(result);
    for (size_t i = 0; i < result.size()-1; i++)
    {
        EXPECT_EQ(1024, result[i]);
    }
}

TEST(gpu_tests, dense_layer1024_batches2_v1)
{
    shape input_shape(1024, 1, 1, 2);
    cuVector<float> input;
    EXPECT_TRUE(input.resize(input_shape.size(), 1.0f));
    shape output_shape(1024, 1, 1, 2);
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size()));
    cuVector<float> weights;
    EXPECT_TRUE(weights.resize(1024 * 1024, 1.0f));
    std::vector<float> result;
    linearLayerForwardPassGPU(output.get(), weights.get(), input.get(), input_shape, output_shape, false);
    output.getCopy(result);
    EXPECT_EQ(result.size(), output_shape.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(1024, result[i]);
    }
}

TEST(gpu_tests, dense_layer1024_batches2_v2)
{
    shape input_shape(1024, 1, 1, 2);
    cuVector<float> input;
    EXPECT_TRUE(input.resize(input_shape.size(), 1.0f));
    shape output_shape(1025, 1, 1, 2);
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size()));
    cuVector<float> weights;
    EXPECT_TRUE(weights.resize(1024 * 1024, 1.0f));
    std::vector<float> result;
    linearLayerForwardPassGPU(output.get(), weights.get(), input.get(), input_shape, shape(1024,1,1,2), true);
    output.getCopy(result);
    EXPECT_EQ(result.size(), output_shape.size());
    for (size_t i = 0; i < 1024; i++)
    {
        EXPECT_EQ(1024, result[i]);
    }
    EXPECT_EQ(0, result[1024]);
    for (size_t i = 1025; i < result.size() - 1; i++)
    {
        EXPECT_EQ(1024, result[i]);
    }
    EXPECT_EQ(0, result[result.size() - 1]);
}

TEST(gpu_tests, test_xor_cpu)
{
    NeuralNet test(2, true);
    test.addLayer(new LinearLayer(2));
    test.addLayer(new SigmoidLayer());
    test.addLayer(new LinearLayer(1));
    test.addLayer(new SigmoidLayer());
    OutputLayer* loss = new OutputLayer();
    test.addLayer(loss);
    test.set_learning_rate(0.01f);
    for (int i = 0; i < 1000000; i++)
    {
        test.getInputLayer().setInput(std::array<float, 2>{0, 1}.data(), 2);
        test.predict();
        loss->setObservedValue({ 1,0 });
        test.backprop();
        test.getInputLayer().setInput(std::array<float, 2>{1, 1}.data(), 2);
        test.predict();
        loss->setObservedValue({ 0,0 });
        test.backprop();
        test.getInputLayer().setInput(std::array<float, 2>{0, 0}.data(), 2);
        test.predict();
        loss->setObservedValue({ 0,0 });
        test.backprop();
        test.getInputLayer().setInput(std::array<float, 2>{1, 0}.data(), 2);
        test.predict();
        loss->setObservedValue({ 1,0 });
        test.backprop();
    }
    test.getInputLayer().setInput(std::array<float, 2>{1, 0}.data(), 2);
    test.predict();
    EXPECT_GT(loss->getOutput()[0], 0.95f);
    EXPECT_EQ(loss->getOutput()[1], 1.0f);
    test.getInputLayer().setInput(std::array<float, 2>{0, 0}.data(), 2);
    test.predict();
    EXPECT_LT(loss->getOutput()[0], 0.05f);
    EXPECT_EQ(loss->getOutput()[1], 1.0f);
    test.getInputLayer().setInput(std::array<float, 2>{1, 1}.data(), 2);
    test.predict();
    EXPECT_LT(loss->getOutput()[0], 0.05f);
    EXPECT_EQ(loss->getOutput()[1], 1.0f);
    test.getInputLayer().setInput(std::array<float, 2>{0, 1}.data(), 2);
    test.predict();
    EXPECT_GT(loss->getOutput()[0], 0.95f);
    EXPECT_EQ(loss->getOutput()[1], 1.0f);
}

TEST(gpu_tests, test_xor_gpu)
{
    NeuralNet test(2, true);
    test.addLayer(new LinearLayerGPU<false>(10));
    test.addLayer(new SigmoidLayerGPU());
    test.addLayer(new LinearLayerGPU<false>(1));
    test.addLayer(new SigmoidLayerGPU());
    OutputLayer* loss = new OutputLayer();
    test.addLayer(loss);
    test.set_learning_rate(0.01f);
    for (int i = 0; i < 10000; i++)
    {
        test.getInputLayer().setInput(std::array<float, 2>{0, 1}.data(), 2);
        test.predict();
        loss->setObservedValue({ 1,0 });
        test.backprop();
        test.getInputLayer().setInput(std::array<float, 2>{1, 1}.data(), 2);
        test.predict();
        loss->setObservedValue({ 0,0 });
        test.backprop();
        test.getInputLayer().setInput(std::array<float, 2>{0, 0}.data(), 2);
        test.predict();
        loss->setObservedValue({ 0,0 });
        test.backprop();
        test.getInputLayer().setInput(std::array<float, 2>{1, 0}.data(), 2);
        test.predict();
        loss->setObservedValue({ 1,0 });
        test.backprop();
    }
    test.getInputLayer().setInput(std::array<float, 2>{1, 0}.data(), 2);
    test.predict();
    EXPECT_GT(loss->getOutput()[0], 0.9f);
    EXPECT_EQ(loss->getOutput()[1], 1.0f);
    test.getInputLayer().setInput(std::array<float, 2>{0, 0}.data(), 2);
    test.predict();
    EXPECT_LT(loss->getOutput()[0], 0.1f);
    EXPECT_EQ(loss->getOutput()[1], 1.0f);
    test.getInputLayer().setInput(std::array<float, 2>{1, 1}.data(), 2);
    test.predict();
    EXPECT_LT(loss->getOutput()[0], 0.1f);
    EXPECT_EQ(loss->getOutput()[1], 1.0f);
    test.getInputLayer().setInput(std::array<float, 2>{0, 1}.data(), 2);
    test.predict();
    EXPECT_GT(loss->getOutput()[0], 0.9f);
    EXPECT_EQ(loss->getOutput()[1], 1.0f);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}