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
    conv2d_kernel(inputK.get(), input_shape, weights.get(), outputK.get(), output_shape, 3);
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
    conv2d_kernel(inputK.get(), input_shape, weights.get(), outputK.get(), output_shape, 3);
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

TEST(gpu_tests, filter_3x3v2_depth3)
{
    shape input_shape;
    input_shape.width = 6;
    input_shape.height = 6;
    input_shape.depth = 3;
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
    conv2d_kernel(inputK.get(), input_shape, weights.get(), outputK.get(), output_shape, 3);
    outputK.getCopy(output);
    std::vector<float> result = { 0,0,0,0,
                                  30,10,-10,-30,
                                  30,10,-10,-30,
                                   0,0,0,0
    };
    EXPECT_EQ(output.size(), result.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(output[i], 3 * result[i]);
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
    conv2d_kernel(inputK.get(), input_shape, weights.get(), outputK.get(), output_shape, 3);
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
        linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr.get(), 1024, 1024);
    }
    outputGPU.getCopy(result);
    for (size_t i = 0; i < result.size()-1; i++)
    {
        EXPECT_EQ(1024, result[i]);
    }
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