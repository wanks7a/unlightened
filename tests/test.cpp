#include "pch.h"
#include "../magic/include/conv_filter.h"
#include "../magic/include/GpuUtils.h"
#include "../magic/include/GpuMemory.h"

class CNN : public ::testing::Test 
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

TEST(CNN, filter_3x3) 
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
    std::cout << "works" << std::endl;
}