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
    input_shape.width = 5;
    input_shape.height = 8;
    std::vector<float> input = { 1,2,3,4,5,
                                 1,2,3,4,5,
                                 1,2,3,4,5,
                                 1,2,3,4,5,
                                 1,2,3,4,5,
                                 1,2,3,4,5,
                                 1,2,3,4,5,
                                 1,2,3,4,5 };
    std::vector<float> output;
    cuVector<float> inputK;
    inputK.setValues(input);
    cuVector<float> outputK;
    output.resize(10 * 7);
    outputK.setValues(output);
    filter_forwardPass(inputK.get(), input_shape, 0, outputK.get());
    outputK.getCopy(output);
}