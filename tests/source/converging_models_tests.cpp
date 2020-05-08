#include <gtest/gtest.h>
#include <NeuralNet.h>
#include <array>
#include <SigmoidLayerGPU.h>
#include <LinearLayerGPU.h>
#include <csv_parser.h>
#include <unordered_map>
#include <cnn_layer.h>
#include <activation_layer.h>
#include <max_pool.h>
#include <chrono>

TEST(model, xor_cpu)
{
    NeuralNet test(shape(3));
    test.addLayer(new LinearLayer(2));
    test.addLayer(new SigmoidLayer());
    test.addLayer(new LinearLayer(1));
    test.addLayer(new SigmoidLayer());
    OutputLayer* loss = new OutputLayer();
    test.addLayer(loss);
    test.set_learning_rate(0.01f);
    for (int i = 0; i < 1000000; i++)
    {
        test.getInputLayer().set_input(std::array<float, 3>{0, 1, 1}.data(), 3);
        test.predict();
        loss->setObservedValue({ 1,0 });
        test.backprop();
        test.getInputLayer().set_input(std::array<float, 3>{1, 1, 1}.data(), 3);
        test.predict();
        loss->setObservedValue({ 0,0 });
        test.backprop();
        test.getInputLayer().set_input(std::array<float, 3>{0, 0, 1}.data(), 3);
        test.predict();
        loss->setObservedValue({ 0,0 });
        test.backprop();
        test.getInputLayer().set_input(std::array<float, 3>{1, 0, 1}.data(), 3);
        test.predict();
        loss->setObservedValue({ 1,0 });
        test.backprop();
    }
    test.getInputLayer().set_input(std::array<float, 3>{1, 0, 1}.data(), 3);
    test.predict();
    EXPECT_GT(loss->get_output()[0], 0.95f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
    test.getInputLayer().set_input(std::array<float, 3>{0, 0, 1}.data(), 3);
    test.predict();
    EXPECT_LT(loss->get_output()[0], 0.05f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
    test.getInputLayer().set_input(std::array<float, 3>{1, 1, 1}.data(), 3);
    test.predict();
    EXPECT_LT(loss->get_output()[0], 0.05f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
    test.getInputLayer().set_input(std::array<float, 3>{0, 1, 1}.data(), 3);
    test.predict();
    EXPECT_GT(loss->get_output()[0], 0.95f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
}

TEST(model, test_xor_gpu)
{
    NeuralNet test(shape(3));
    test.addLayer(new LinearLayerGPU(10));
    test.addLayer(new SigmoidLayerGPU());
    test.addLayer(new LinearLayerGPU(1));
    test.addLayer(new SigmoidLayerGPU());
    OutputLayer* loss = new OutputLayer();
    test.addLayer(loss);
    test.set_learning_rate(0.1f);
    for (int i = 0; i < 10000; i++)
    {
        test.getInputLayer().set_input(std::array<float, 3>{0, 1, 1}.data(), 3);
        test.predict();
        loss->setObservedValue({ 1,0 });
        test.backprop();
        test.getInputLayer().set_input(std::array<float, 3>{1, 1, 1}.data(), 3);
        test.predict();
        loss->setObservedValue({ 0,0 });
        test.backprop();
        test.getInputLayer().set_input(std::array<float, 3>{0, 0, 1}.data(), 3);
        test.predict();
        loss->setObservedValue({ 0,0 });
        test.backprop();
        test.getInputLayer().set_input(std::array<float, 3>{1, 0, 1}.data(), 3);
        test.predict();
        loss->setObservedValue({ 1,0 });
        test.backprop();
    }
    test.getInputLayer().set_input(std::array<float, 3>{1, 0, 1}.data(), 3);
    test.predict();
    EXPECT_GT(loss->get_output()[0], 0.9f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
    test.getInputLayer().set_input(std::array<float, 3>{0, 0, 1}.data(), 3);
    test.predict();
    EXPECT_LT(loss->get_output()[0], 0.1f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
    test.getInputLayer().set_input(std::array<float, 3>{1, 1, 1}.data(), 3);
    test.predict();
    EXPECT_LT(loss->get_output()[0], 0.1f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
    test.getInputLayer().set_input(std::array<float, 3>{0, 1, 1}.data(), 3);
    test.predict();
    EXPECT_GT(loss->get_output()[0], 0.9f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
}

TEST(model, cnn_converge)
{
    NeuralNet test(shape(2,2,1,2));
    auto cnn = new cnn_layer(2, 1);
    test.addLayer(new cnn_layer(2, 1));
    test.addLayer(new activation_layer(activation_layer::activation_function::ReLU)); 
    OutputLayer* loss = new OutputLayer();
    test.addLayer(loss);
    test.set_learning_rate(0.1f);

    for (size_t i = 0; i < 1000; i++)
    {
        test.getInputLayer().set_input({ 0, 1,
                                         1, 0,
                                         1, 0,
                                         0, 1 });
        test.predict();
        loss->setObservedValue({ 1, 0 });
        test.backprop();
    }
    test.getInputLayer().set_input({ 1, 0,
                                     0, 1,
                                     0, 1,
                                     1, 0,
                                     });
    test.predict();
    loss->print_predicted();
}