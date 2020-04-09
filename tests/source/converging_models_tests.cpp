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

TEST(model, mnist_digits)
{
    //csv<float> mnist("C:\\Users\\wanks7a\\Desktop\\mnist_test.csv");

    //for (size_t i = 0; i < mnist.rows.size(); i++)
    //{
    //    for (size_t j = 1; j < mnist.rows[i].elements.size(); j++)
    //    {
    //        mnist.rows[i].elements[j] = mnist.rows[i].elements[j] / 255;
    //    }
    //}

    //std::unordered_map<float, std::vector<float>> possible_outputs =
    //{
    //    {0.0f, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1} },
    //    {1.0f, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1} },
    //    {2.0f, {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1} },
    //    {3.0f, {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1} },
    //    {4.0f, {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1} },
    //    {5.0f, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1} },
    //    {6.0f, {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1} },
    //    {7.0f, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1} },
    //    {8.0f, {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1} },
    //    {9.0f, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1} },
    //};
    //const size_t batch_size = 128;
    //shape mnist_shape(28, 28, 1, batch_size);
    //NeuralNet net(mnist_shape);
    //net.addLayer(new cnn_layer(3, 32));
    //net.addLayer(new activation_layer(activation_layer::activation_function::Sigmoid)); // TODO replace sigmoid with RELU
    //net.addLayer(new cnn_layer(3, 64));
    //net.addLayer(new activation_layer(activation_layer::activation_function::Sigmoid)); // TODO replace sigmoid with RELU
    //net.addLayer(new max_pool(2));
    //net.addLayer(new cnn_layer(3, 1));
    ////net.addLayer(new LinearLayerGPU(128));
    ////net.addLayer(new activation_layer(activation_layer::activation_function::Sigmoid)); // TODO replace sigmoid with RELU
    //net.addLayer(new LinearLayerGPU(10));
    //net.addLayer(new activation_layer(activation_layer::activation_function::Sigmoid)); // TODO replace sigmoid with RELU
    //net.addLayer(new LinearLayerGPU(10));
    //net.addLayer(new activation_layer(activation_layer::activation_function::Sigmoid)); // TODO replace sigmoid with RELU
    //OutputLayer* loss = new OutputLayer();
    //net.addLayer(loss);
    //net.set_learning_rate(0.98f);
    //std::vector<float> one_batch(mnist_shape.size());
    //std::vector<float> one_batch_results((size_t)(11 * batch_size));
    //size_t batch_counter = mnist.rows.size() - (mnist.rows.size() % batch_size);
    //for (size_t k = 0; k < 12; k++)
    //{
    //    for (size_t i = 0; i < batch_counter; i += batch_size)
    //    {
    //        for (size_t j = 0; j < batch_size; j++)
    //        {
    //            memcpy(&one_batch[j * mnist_shape.volume()], mnist.rows[i + j].elements.data() + 1, mnist_shape.volume() * sizeof(float));
    //            memcpy(&one_batch_results[j * 11], possible_outputs[mnist.rows[i + j].elements[0]].data(), 11 * sizeof(float));
    //        }
    //        net.getInputLayer().set_input(one_batch);
    //        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    //        net.predict();
    //        loss->setObservedValue(one_batch_results);
    //        net.backprop();
    //        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    //    }
    //}
}

TEST(model, xor_cpu)
{
    //NeuralNet test(shape(3));
    //test.addLayer(new LinearLayer(2));
    //test.addLayer(new SigmoidLayer());
    //test.addLayer(new LinearLayer(1));
    //test.addLayer(new SigmoidLayer());
    //OutputLayer* loss = new OutputLayer();
    //test.addLayer(loss);
    //test.set_learning_rate(0.01f);
    //for (int i = 0; i < 1000000; i++)
    //{
    //    test.getInputLayer().set_input(std::array<float, 3>{0, 1, 1}.data(), 3);
    //    test.predict();
    //    loss->setObservedValue({ 1,0 });
    //    test.backprop();
    //    test.getInputLayer().set_input(std::array<float, 3>{1, 1, 1}.data(), 3);
    //    test.predict();
    //    loss->setObservedValue({ 0,0 });
    //    test.backprop();
    //    test.getInputLayer().set_input(std::array<float, 3>{0, 0, 1}.data(), 3);
    //    test.predict();
    //    loss->setObservedValue({ 0,0 });
    //    test.backprop();
    //    test.getInputLayer().set_input(std::array<float, 3>{1, 0, 1}.data(), 3);
    //    test.predict();
    //    loss->setObservedValue({ 1,0 });
    //    test.backprop();
    //}
    //test.getInputLayer().set_input(std::array<float, 3>{1, 0, 1}.data(), 3);
    //test.predict();
    //EXPECT_GT(loss->get_output()[0], 0.95f);
    //EXPECT_EQ(loss->get_output()[1], 1.0f);
    //test.getInputLayer().set_input(std::array<float, 3>{0, 0, 1}.data(), 3);
    //test.predict();
    //EXPECT_LT(loss->get_output()[0], 0.05f);
    //EXPECT_EQ(loss->get_output()[1], 1.0f);
    //test.getInputLayer().set_input(std::array<float, 3>{1, 1, 1}.data(), 3);
    //test.predict();
    //EXPECT_LT(loss->get_output()[0], 0.05f);
    //EXPECT_EQ(loss->get_output()[1], 1.0f);
    //test.getInputLayer().set_input(std::array<float, 3>{0, 1, 1}.data(), 3);
    //test.predict();
    //EXPECT_GT(loss->get_output()[0], 0.95f);
    //EXPECT_EQ(loss->get_output()[1], 1.0f);
}

TEST(model, test_xor_gpu)
{
    //NeuralNet test(shape(3));
    //test.addLayer(new LinearLayerGPU(10));
    //test.addLayer(new SigmoidLayerGPU());
    //test.addLayer(new LinearLayerGPU(1));
    //test.addLayer(new SigmoidLayerGPU());
    //OutputLayer* loss = new OutputLayer();
    //test.addLayer(loss);
    //test.set_learning_rate(0.01f);
    //for (int i = 0; i < 10000; i++)
    //{
    //    test.getInputLayer().set_input(std::array<float, 3>{0, 1, 1}.data(), 3);
    //    test.predict();
    //    loss->setObservedValue({ 1,0 });
    //    test.backprop();
    //    test.getInputLayer().set_input(std::array<float, 3>{1, 1, 1}.data(), 3);
    //    test.predict();
    //    loss->setObservedValue({ 0,0 });
    //    test.backprop();
    //    test.getInputLayer().set_input(std::array<float, 3>{0, 0, 1}.data(), 3);
    //    test.predict();
    //    loss->setObservedValue({ 0,0 });
    //    test.backprop();
    //    test.getInputLayer().set_input(std::array<float, 3>{1, 0, 1}.data(), 3);
    //    test.predict();
    //    loss->setObservedValue({ 1,0 });
    //    test.backprop();
    //}
    //test.getInputLayer().set_input(std::array<float, 3>{1, 0, 1}.data(), 3);
    //test.predict();
    //EXPECT_GT(loss->get_output()[0], 0.9f);
    //EXPECT_EQ(loss->get_output()[1], 1.0f);
    //test.getInputLayer().set_input(std::array<float, 3>{0, 0, 1}.data(), 3);
    //test.predict();
    //EXPECT_LT(loss->get_output()[0], 0.1f);
    //EXPECT_EQ(loss->get_output()[1], 1.0f);
    //test.getInputLayer().set_input(std::array<float, 3>{1, 1, 1}.data(), 3);
    //test.predict();
    //EXPECT_LT(loss->get_output()[0], 0.1f);
    //EXPECT_EQ(loss->get_output()[1], 1.0f);
    //test.getInputLayer().set_input(std::array<float, 3>{0, 1, 1}.data(), 3);
    //test.predict();
    //EXPECT_GT(loss->get_output()[0], 0.9f);
    //EXPECT_EQ(loss->get_output()[1], 1.0f);
}

TEST(model, cnn_converge)
{
    NeuralNet test(shape(2,2));
    test.addLayer(new cnn_layer(2, 1));
    test.addLayer(new activation_layer(activation_layer::activation_function::Sigmoid)); 
    OutputLayer* loss = new OutputLayer();
    test.addLayer(loss);
    test.set_learning_rate(0.3f);

    for (size_t i = 0; i < 10000; i++)
    {
        test.getInputLayer().set_input({1, 0,
                                        0, 1});
        test.predict();
        loss->setObservedValue({ 1 });
        test.backprop();
        test.getInputLayer().set_input({ 0, 1,
                                         1, 0 });
        test.predict();
        loss->setObservedValue({ 0 });
        test.backprop();
    }
    test.getInputLayer().set_input({ 1, 0,
                                0, 1 });
    test.predict();
    loss->setObservedValue({ 1 });
    test.backprop();
    test.getInputLayer().set_input({ 0, 1,
                                     1, 0 });
    test.predict();
    loss->setObservedValue({ 0 });
    test.backprop();
}