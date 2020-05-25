
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
#include <loss_plot.h>
#include <reshape_layer.h>
#include <gan.h>

void startTrain(shape_plot* plot, loss_plot* l_plot)
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
            l_plot->add_point(loss->get_total_loss(), "loss" , "r");
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << " batch  = " << k << " sample = " << i / batch_size << std::endl;
        }
    }
}

void gan_test(shape_plot* sh_plot, loss_plot* l_plot)
{
    csv<float> mnist("C:\\Users\\wanks7a\\Desktop\\mnist_test.csv");
    std::vector<float> fake_data;
    const size_t batch_size = 64;
    shape gen_shape(128, 1, 1, batch_size);

    std::thread t([&]() {
        size_t currentSize = mnist.rows.size() * gen_shape.area();
        fake_data.reserve(currentSize);
        std::mt19937_64 rng;
        // initialize the random number generator with time-dependent seed
        uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::seed_seq ss{ uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32) };
        rng.seed(ss);

        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

        for (int i = 0; i < currentSize; i++)
        {
            fake_data.emplace_back(distribution(rng));
        }
        });

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
    shape mnist_shape(28, 28, 1, batch_size);
    NeuralNet* discriminator = new NeuralNet(mnist_shape);
    discriminator->addLayer(new cnn_layer(3, 32));
    discriminator->addLayer(new activation_layer(activation_layer::activation_function::ReLU));
    discriminator->addLayer(new cnn_layer(3, 64));
    discriminator->addLayer(new activation_layer(activation_layer::activation_function::ReLU));
    discriminator->addLayer(new cnn_layer(3, 64));
    discriminator->addLayer(new activation_layer(activation_layer::activation_function::ReLU));
    discriminator->addLayer(new LinearLayerGPU(128));
    discriminator->addLayer(new activation_layer(activation_layer::activation_function::ReLU));
    discriminator->addLayer(new LinearLayerGPU(2, false));
    discriminator->addLayer(new activation_layer(activation_layer::activation_function::Softmax));

    NeuralNet* generator = new NeuralNet(gen_shape);
    generator->addLayer(new LinearLayerGPU(28*28, false));
    generator->addLayer(new reshape_layer(shape(28,28,1,batch_size)));
    //cnn_layer* cnn_3_64 = new cnn_layer(3, 64);
    //filter_options opt = cnn_3_64->get_options();
    //opt.zeropadding = true;
    //cnn_3_64->set_options(opt);
    //generator->addLayer(cnn_3_64);
    //generator->addLayer(new activation_layer(activation_layer::activation_function::ReLU));
    //cnn_layer* cnn_3_128 = new cnn_layer(3, 128);
    //opt = cnn_3_128->get_options();
    //opt.zeropadding = true;
    //cnn_3_128->set_options(opt);
    //generator->addLayer(cnn_3_128);
    //generator->addLayer(new activation_layer(activation_layer::activation_function::ReLU));
    cnn_layer* cnn_3_1 = new cnn_layer(3, 1);
    auto opt = cnn_3_1->get_options();
    opt.zeropadding = true;
    cnn_3_1->set_options(opt);
    generator->addLayer(cnn_3_1);
    generator->addLayer(new activation_layer(activation_layer::activation_function::Sigmoid));

    generator->set_learning_rate(0.3f);
    discriminator->set_learning_rate(0.01f);
    gan g;
    g.set_discriminator(discriminator);
    g.set_generator(generator);
    t.join();
    std::vector<float> one_batch(mnist_shape.size());
    std::vector<float> gen_noise(gen_shape.size());
    size_t batch_counter = mnist.rows.size() - (mnist.rows.size() % batch_size);

    for (size_t k = 0; k < 10000; k++)
    {
        double total_loss_generator = 0.0;
        double total_loss_disc = 0.0;
        for (size_t i = 0; i < batch_counter; i += batch_size)
        {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            for (size_t j = 0; j < batch_size; j++)
            {
                memcpy(&one_batch[j * mnist_shape.volume()], mnist.rows[i + j].elements.data() + 1, mnist_shape.volume() * sizeof(float));
            }
            memcpy(gen_noise.data(), &fake_data[i * gen_shape.area()], gen_shape.size());
            g.predict(one_batch, gen_noise);
            sh_plot->set_shape_data((*generator)[3]->get_native_output(), shape(28,28,1,batch_size));
            g.backprop();
            total_loss_disc += g.discriminator_loss() / batch_size;
            total_loss_generator += g.generator_loss() / batch_size;
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << " batch  = " << k << " sample = " << i / batch_size << std::endl;
        }
        l_plot->add_point(total_loss_disc, "discrminator", "r");
        l_plot->add_point(total_loss_generator, "generator", "b");
    }
}

int main(int argc, char* argv[])
{
    viewer view_manager;
    shape_plot* sh_plot = new shape_plot();
    loss_plot* l_plot = new loss_plot();
    view_manager.add_view(sh_plot);
    view_manager.add_view(l_plot);
    std::thread t([&]() {
        //startTrain(sh_plot, l_plot);
        gan_test(sh_plot, l_plot);
        });
    view_manager.loop();
	return 0;
}