#include <gtest/gtest.h>
#include <NeuralNet.h>
#include <array>
#include <SigmoidLayerGPU.h>
#include <LinearLayerGPU.h>
#include <csv_parser.h>
#include <unordered_map>
#include <activation_layer.h>
#include <max_pool.h>
#include <chrono>

//TEST(model, xor_cpu)
//{
//    model test(shape(3));
//    test.addLayer(new dense_layer(2));
//    test.addLayer(new SigmoidLayer());
//    test.addLayer(new dense_layer(1));
//    test.addLayer(new SigmoidLayer());
//    test.set_learning_rate(0.01f);
//    for (int i = 0; i < 1000000; i++)
//    {
//        test.getInputLayer().set_input(std::array<float, 3>{0, 1, 1}.mem(), 3);
//        test.predict();
//        test.loss_layer().setObservedValue({ 1,0 });
//        test.backprop();
//        test.getInputLayer().set_input(std::array<float, 3>{1, 1, 1}.mem(), 3);
//        test.predict();
//        test.loss_layer().setObservedValue({ 0,0 });
//        test.backprop();
//        test.getInputLayer().set_input(std::array<float, 3>{0, 0, 1}.mem(), 3);
//        test.predict();
//        test.loss_layer().setObservedValue({ 0,0 });
//        test.backprop();
//        test.getInputLayer().set_input(std::array<float, 3>{1, 0, 1}.mem(), 3);
//        test.predict();
//        test.loss_layer().setObservedValue({ 1,0 });
//        test.backprop();
//    }
//    test.getInputLayer().set_input(std::array<float, 3>{1, 0, 1}.mem(), 3);
//    test.predict();
//    EXPECT_GT(test.loss_layer().get_output()[0], 0.95f);
//    EXPECT_EQ(test.loss_layer().get_output()[1], 1.0f);
//    test.getInputLayer().set_input(std::array<float, 3>{0, 0, 1}.mem(), 3);
//    test.predict();
//    EXPECT_LT(test.loss_layer().get_output()[0], 0.05f);
//    EXPECT_EQ(test.loss_layer().get_output()[1], 1.0f);
//    test.getInputLayer().set_input(std::array<float, 3>{1, 1, 1}.mem(), 3);
//    test.predict();
//    EXPECT_LT(test.loss_layer().get_output()[0], 0.05f);
//    EXPECT_EQ(test.loss_layer().get_output()[1], 1.0f);
//    test.getInputLayer().set_input(std::array<float, 3>{0, 1, 1}.mem(), 3);
//    test.predict();
//    EXPECT_GT(test.loss_layer().get_output()[0], 0.95f);
//    EXPECT_EQ(test.loss_layer().get_output()[1], 1.0f);
//}
//
//TEST(model, test_xor_gpu)
//{
//    model test(shape(2));
//    test.addLayer(new dense_gpu(10));
//    test.addLayer(new activation_layer(activation_layer::activation_function::Sigmoid));
//    test.addLayer(new dense_gpu(1));
//    test.addLayer(new activation_layer(activation_layer::activation_function::Sigmoid));
//    test.set_learning_rate(0.1f);
//    for (int i = 0; i < 10000; i++)
//    {
//        test.getInputLayer().set_input(std::array<float, 2>{0, 1}.mem(), 2);
//        test.predict();
//        test.loss_layer().setObservedValue({ 1 });
//        test.backprop();
//        test.getInputLayer().set_input(std::array<float, 2>{1, 1}.mem(), 2);
//        test.predict();
//        test.loss_layer().setObservedValue({ 0 });
//        test.backprop();
//        test.getInputLayer().set_input(std::array<float, 2>{0, 0}.mem(), 2);
//        test.predict();
//        test.loss_layer().setObservedValue({ 0 });
//        test.backprop();
//        test.getInputLayer().set_input(std::array<float, 2>{1, 0}.mem(), 2);
//        test.predict();
//        test.loss_layer().setObservedValue({ 1 });
//        test.backprop();
//    }
//    test.getInputLayer().set_input(std::array<float, 2>{1, 0}.mem(), 2);
//    test.predict();
//    EXPECT_GT(test.loss_layer().get_output()[0], 0.9f);
//    test.getInputLayer().set_input(std::array<float, 2>{0, 0}.mem(), 2);
//    test.predict();
//    EXPECT_LT(test.loss_layer().get_output()[0], 0.1f);
//    test.getInputLayer().set_input(std::array<float, 2>{1, 1}.mem(), 2);
//    test.predict();
//    EXPECT_LT(test.loss_layer().get_output()[0], 0.1f);
//    test.getInputLayer().set_input(std::array<float, 2>{0, 1}.mem(), 2);
//    test.predict();
//    EXPECT_GT(test.loss_layer().get_output()[0], 0.9f);
//}
//
//TEST(model, cnn_converge)
//{
//    model test(shape(2,2,1,2));
//    auto cnn = new cnn_layer(2, 1);
//    test.addLayer(new cnn_layer(2, 1));
//    test.addLayer(new activation_layer(activation_layer::activation_function::ReLU));
//    test.set_learning_rate(0.1f);
//
//    for (size_t i = 0; i < 1000; i++)
//    {
//        test.getInputLayer().set_input({ 0, 1,
//                                         1, 0,
//                                         1, 0,
//                                         0, 1 });
//        test.predict();
//        test.loss_layer().setObservedValue({ 1, 0 });
//        test.backprop();
//    }
//    test.getInputLayer().set_input({ 1, 0,
//                                     0, 1,
//                                     0, 1,
//                                     1, 0,
//                                     });
//    test.predict();
//    test.loss_layer().print_predicted();
//}