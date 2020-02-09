#include "pch.h"
#include "../magic/include/conv_filter.h"
#include "../magic/include/GpuUtils.h"
#include "../magic/include/GpuMemory.h"
#include "../magic/include/LinearLayerGPU.h"

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

TEST(gpu_tests, filter_3x3v1)
{
    filter_options opt(2, 2);
    filter_conv2d filter(opt);
    shape input_shape;
    input_shape.width = 7;
    input_shape.height = 7;
    shape output_shape;
    output_shape.width = 5;
    output_shape.height = 5;
    std::vector<float> input = { 
                                 0,0,0,0,0,0,0,
                                 0,2,2,2,1,2,0,
                                 0,0,2,1,2,2,0,
                                 0,1,1,0,1,0,0,
                                 0,2,1,1,1,2,0,
                                 0,1,1,2,2,2,0,
                                 0,0,0,0,0,0,0
                               };
    cuVector<float> weights;
    weights.setValues({ 
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1
                      });
    std::vector<float> output;
    cuVector<float> inputK;
    inputK.setValues(input);
    cuVector<float> outputK;
    output.resize(static_cast<size_t>(output_shape.width) * output_shape.height);
    outputK.setValues(output);
    filter_forwardPass(inputK.get(), input_shape, weights.get(), outputK.get(), output_shape, 3);
    outputK.getCopy(output);
    std::vector<float> result = { 0, 1, 2, 2, 2,
                                  -4, -4, -4, -4, -2,
                                    0, -1, -2, -1, 0,
                                    -1, 1, 2, 1, 2,
                                     -4, -4, -4, -5, -2 };
    EXPECT_EQ(output.size(), result.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(output[i], result[i]);
    }
}

TEST(gpu_tests, filter_3x3v2)
{
    shape input_shape;
    input_shape.width = 6;
    input_shape.height = 6;
    shape output_shape;
    output_shape.width = 4;
    output_shape.height = 4;
    std::vector<float> input = {
                                 10,10,10,0,0,0,
                                 10,10,10,0,0,0,
                                 10,10,10,0,0,0,
                                 0,0,0,10,10,10,
                                 0,0,0,10,10,10,
                                 0,0,0,10,10,10,
    };
    cuVector<float> weights;
    weights.setValues({
                        1,1,1,
                        0,0,0,
                        -1,-1,-1
        });
    std::vector<float> output;
    cuVector<float> inputK;
    inputK.setValues(input);
    cuVector<float> outputK;
    output.resize(static_cast<size_t>(output_shape.width)* output_shape.height);
    outputK.setValues(output);
    filter_forwardPass(inputK.get(), input_shape, weights.get(), outputK.get(), output_shape, 3);
    outputK.getCopy(output);
    std::vector<float> result = { 0,0,0,0,
                                  30,10,-10,-30,
                                  30,10,-10,-30,
                                   0,0,0,0
    };
    EXPECT_EQ(output.size(), result.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(output[i], result[i]);
    }
}

TEST(gpu_tests, filter_3x3v3)
{
    shape input_shape;
    input_shape.width = 7;
    input_shape.height = 7;
    shape output_shape;
    output_shape.width = 5;
    output_shape.height = 5;
    std::vector<float> input = {
                                 0,1,1,1,0,0,0,
                                 0,0,1,1,1,0,0,
                                 0,0,0,1,1,1,0,
                                 0,0,0,1,1,0,0,
                                 0,0,1,1,0,0,0,
                                 0,1,1,0,0,0,0,
                                 1,1,0,0,0,0,0
    };
    cuVector<float> weights;
    weights.setValues({
                        1,0,1,
                        0,1,0,
                        1,0,1
        });
    std::vector<float> output;
    cuVector<float> inputK;
    inputK.setValues(input);
    cuVector<float> outputK;
    output.resize(static_cast<size_t>(output_shape.width)* output_shape.height);
    outputK.setValues(output);
    filter_forwardPass(inputK.get(), input_shape, weights.get(), outputK.get(), output_shape, 3);
    outputK.getCopy(output);
    std::vector<float> result = { 1,4,3,4,1,
                                  1,2,4,3,3,
                                  1,2,3,4,1,
                                  1,3,3,1,1,
                                  3,3,1,1,0
    };
    EXPECT_EQ(output.size(), result.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(output[i], result[i]);
    }
}

TEST(gpu_tests, dense_layer1024)
{
    std::vector<float> input;
    input.resize(1024 * 1024, 1.0f);
    LinearLayerGPU<false> l(1024);
    l.setInputSize(1024);
    l.init();
    EXPECT_EQ(l.getOutputSize(), 1024 + 1); // because of bias
    EXPECT_EQ(l.getInputSize(), 1024);
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
        linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr.get(), 1024, 1024);
    }
    outputGPU.getCopy(result);
}

