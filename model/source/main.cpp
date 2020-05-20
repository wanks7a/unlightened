
#include <AllLayers.h>
#include <NeuralNet.h>
#include <cnn_layer.h>
#include <unordered_map>
#include <vector>
#include <csv_parser.h>
#include <activation_layer.h>
#include <max_pool.h>
#include <LinearLayerGPU.h>
#include <mgl2/mgl.h>
#include <iostream>
#include <SDL.h>
#include <SDL_video.h>
#include <shape_plot.h>

void startTrain(shape_plot* plot)
{
    csv<float> mnist("C:\\Users\\wanks7a\\Desktop\\mnist_test.csv");

    for (size_t i = 0; i < mnist.rows.size(); i++)
    {
        for (size_t j = 1; j < mnist.rows[i].elements.size(); j++)
        {
            mnist.rows[i].elements[j] = mnist.rows[i].elements[j] / 255;
        }
    }

    std::unordered_map<float, std::vector<float>> possible_outputs =
    {
        {0.0f, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0} },
        {1.0f, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0} },
        {2.0f, {0, 0, 1, 0, 0, 0, 0, 0, 0, 0} },
        {3.0f, {0, 0, 0, 1, 0, 0, 0, 0, 0, 0} },
        {4.0f, {0, 0, 0, 0, 1, 0, 0, 0, 0, 0} },
        {5.0f, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0} },
        {6.0f, {0, 0, 0, 0, 0, 0, 1, 0, 0, 0} },
        {7.0f, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0} },
        {8.0f, {0, 0, 0, 0, 0, 0, 0, 0, 1, 0} },
        {9.0f, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1} },
    };
    const size_t batch_size = 128;
    shape mnist_shape(28, 28, 1, batch_size);
    NeuralNet net(mnist_shape);
    net.addLayer(new cnn_layer(3, 32));
    net.addLayer(new activation_layer(activation_layer::activation_function::ReLU));
    net.addLayer(new cnn_layer(3, 64));
    net.addLayer(new activation_layer(activation_layer::activation_function::ReLU));
    net.addLayer(new max_pool(2));
    net.addLayer(new LinearLayerGPU(128));
    net.addLayer(new activation_layer(activation_layer::activation_function::ReLU));
    net.addLayer(new LinearLayerGPU(10, false));
    net.addLayer(new activation_layer(activation_layer::activation_function::Softmax));
    OutputLayer* loss = new OutputLayer();
    net.addLayer(loss);
    float l_rate = 0.1f;
    net.set_learning_rate(l_rate);

    std::vector<float> one_batch(mnist_shape.size());
    std::vector<float> one_batch_results((size_t)(10 * batch_size));
    size_t batch_counter = mnist.rows.size() - (mnist.rows.size() % batch_size);
    for (size_t k = 0; k < 12; k++)
    {
        for (size_t i = 0; i < batch_counter; i += batch_size)
        {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            for (size_t j = 0; j < batch_size; j++)
            {
                memcpy(&one_batch[j * mnist_shape.volume()], mnist.rows[i + j].elements.data() + 1, mnist_shape.volume() * sizeof(float));
                memcpy(&one_batch_results[j * 10], possible_outputs[mnist.rows[i + j].elements[0]].data(), 10 * sizeof(float));
            }
            net.getInputLayer().set_input(one_batch);

            net.predict();
            loss->setObservedValue(one_batch_results);
            net.backprop();
            loss->print_predicted(10);
            auto l = net[4]->get_native_output();
            auto shape = net[4]->get_shape();
            plot->set_shape_data(std::move(l), shape);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << " batch  = " << k << " sample = " << i / batch_size << std::endl;
        }
    }
    net.getInputLayer().set_input(one_batch);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    net.predict();
    loss->setObservedValue(one_batch_results);
    net.getInputLayer().set_input(one_batch);
    begin = std::chrono::steady_clock::now();
    net.predict();
    loss->setObservedValue(one_batch_results);
    net.getInputLayer().set_input(one_batch);
    begin = std::chrono::steady_clock::now();
    net.predict();
    loss->setObservedValue(one_batch_results);
    net.getInputLayer().set_input(one_batch);
    begin = std::chrono::steady_clock::now();
    net.predict();
    loss->setObservedValue(one_batch_results);
    net.getInputLayer().set_input(one_batch);
    begin = std::chrono::steady_clock::now();
    net.predict();
    loss->setObservedValue(one_batch_results);
}

int main(int argc, char* argv[])
{
    viewer view_manager;
    shape_plot* sh_plot = new shape_plot();
    view_manager.add_view(sh_plot);
    std::thread t([&]() {
        startTrain(sh_plot);
        });
    view_manager.loop();
	return 0;
}