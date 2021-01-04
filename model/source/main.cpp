
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
#include <conv2d_cudnn.h>
#include <conv_transpose.h>
#include <fstream>
#include <map>
#include <image_helper.h>
#include <optimizer_momentum.h>
#include <batch_norm_cuda.h>

class std_stream : public generic_stream
{
public:
    std::vector<char> data;
    std::string file_name;

    std_stream(const std::string& ptr) : file_name(ptr) {};

    size_t write(const char* ptr, size_t bytes) override
    {
        for (size_t i = 0; i < bytes; i++)
        {
            data.emplace_back(ptr[i]);
        }
        return bytes;
    }

    size_t read(char* buff, size_t bytes) override
    {
        size_t result = peek(buff, bytes);
        data.erase(data.begin(), data.begin() + result);
        return result;
    }

    size_t peek(char* buff, size_t bytes) const override
    {
        size_t result;
        if (data.size() < bytes)
        {
            result = data.size();
        }
        else
        {
            result = bytes;
        }

        for (size_t i = 0; i < result; i++)
        {
            buff[i] = data[i];
        }

        return result;
    }

    void save()
    {
        std::ofstream outfile(file_name.c_str(), std::ofstream::binary | std::ofstream::trunc);   // get the size
        outfile.write(data.data(), data.size());                      // write the actual text
        outfile.close();
        data.clear();
    }

    void read()
    {
        std::ifstream input(file_name.c_str(), std::ios::binary);

        // copies all data into buffer
        std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});
        data = std::move(buffer);
    }

    bool is_open() const override
    {
        std::ifstream input(file_name.c_str(), std::ios::binary);
        return input.is_open();
    }
};

void startTrain(shape_plot* plot, loss_plot* l_plot)
{
    csv<float> mnist("C:\\Users\\wanks7a\\Desktop\\mnist_train.csv");

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
    const size_t batch_size = 256;
    shape mnist_shape(28, 28, 1, batch_size);
    model net(mnist_shape);
    net.addLayer(new conv2d_cudnn(3, 32));
    net.addLayer(new activation_layer(activation_layer::activation_function::ReLU));
    net.addLayer(new max_pool(2));
    net.addLayer(new conv2d_cudnn(3, 64));
    net.addLayer(new activation_layer(activation_layer::activation_function::ReLU));
    net.addLayer(new max_pool(2));
    net.addLayer(new dense_gpu(128));
    net.addLayer(new activation_layer(activation_layer::activation_function::ReLU));
    net.addLayer(new dense_gpu(10));
    net.addLayer(new activation_layer(activation_layer::activation_function::Softmax));
    net.set_optimizer<momentum_optimizer>(0.9f);
    net.loss_layer().set_loss_func(LOSS::binary_cross_entropy);
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
            net.loss_layer().set_observed(one_batch_results);
            net.backprop();
            net.loss_layer().print_predicted(1);
            auto l = net[4]->get_native_output();
            auto shape = net[4]->get_shape();
            l_plot->add_point(net.loss_layer().get_mean_loss(), "loss" , "r");
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << " batch  = " << k << " sample = " << i / batch_size << std::endl;
        }
    }
}

void gan_test(shape_plot* sh_plot, loss_plot* l_plot)
{
    csv<float> mnist("C:\\Users\\wanks7a\\Desktop\\mnist_train.csv");
    std::vector<float> fake_data;
    const size_t batch_size = 256;
    shape gen_shape(100, 1, 1, batch_size);

    std::thread t([&]() {
        size_t currentSize = mnist.rows.size() * gen_shape.area();
        fake_data.reserve(currentSize);
        std::mt19937_64 rng;
        // initialize the random number generator with time-dependent seed
        uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::seed_seq ss{ uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32) };
        rng.seed(ss);

        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

        for (int i = 0; i < currentSize; i++)
        {
            fake_data.emplace_back(distribution(rng));
        }
        });

    for (size_t i = 0; i < mnist.rows.size(); i++)
    {
        for (size_t j = 1; j < mnist.rows[i].elements.size(); j++)
        {
            mnist.rows[i].elements[j] = (mnist.rows[i].elements[j] - 127.5) / 127.5;
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
    model* discriminator = new model(mnist_shape);
    filter_options opt;
    opt.w = 5;
    opt.h = 5;
    opt.zeropadding = true;
    opt.num_of_filters = 64;
    opt.stride = 2;
    discriminator->addLayer(new conv2d_cudnn(opt));
    discriminator->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));
    opt.num_of_filters = 128;
    discriminator->addLayer(new conv2d_cudnn(opt));
    discriminator->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));
    discriminator->addLayer(new dense_gpu(1));
    discriminator->addLayer(new activation_layer(activation_layer::activation_function::Sigmoid));
    model* generator = new model(gen_shape);
    //generator->addLayer(new dense_gpu(30 * 30));
    //generator->addLayer(new reshape_layer(shape(30, 30)));
    //generator->addLayer(new activation_layer(activation_layer::activation_function::ReLU));
    //generator->addLayer(new conv2d_cudnn(2, 32));
    //generator->addLayer(new activation_layer(activation_layer::activation_function::ReLU));
    //generator->addLayer(new conv2d_cudnn(2, 1));
    //generator->addLayer(new activation_layer(activation_layer::activation_function::Sigmoid));
    generator->addLayer(new dense_gpu(7*7, false));
    generator->addLayer(new reshape_layer(shape(7, 7)));
    generator->addLayer(new batch_norm_cuda());
    generator->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));
    generator->addLayer(new conv2d_transposed(128,5,1, conv2d_transposed::padding::SAME));
    generator->addLayer(new batch_norm_cuda());
    generator->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));
    generator->addLayer(new conv2d_transposed(128, 5, 2, conv2d_transposed::padding::SAME));
    generator->addLayer(new batch_norm_cuda());
    generator->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));
    generator->addLayer(new conv2d_transposed(1, 5, 2, conv2d_transposed::padding::SAME));
    generator->addLayer(new activation_layer(activation_layer::activation_function::tanh));
    auto s1 = std::make_shared<std_stream>("disc.b");
    auto s2 = std::make_shared<std_stream>("gen.b");
      
    binary_serialization dis_ser(s1);
    binary_serialization gen_ser(s2);
    s1->read();
    s2->read();
    dis_ser.deserialize_model(*discriminator);
    gen_ser.deserialize_model(*generator);
    generator->set_learning_rate(0.1f);
    discriminator->set_learning_rate(0.1f);
    discriminator->set_optimizer<momentum_optimizer>(0.1f);
    generator->set_optimizer<momentum_optimizer>(0.1f);


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
            memcpy(gen_noise.data(), &fake_data[i * gen_shape.area()], gen_shape.size() * sizeof(float));
            g.predict(one_batch, gen_noise);
            std::vector<float> image = (*generator)[11]->get_native_output();
            image.resize(28 * 28);
            sh_plot->set_scale(8.0f);
            sh_plot->draw_grayscale(std::move(image), 28, 28, -1.0f, 1.0f);
            g.backprop();
            total_loss_disc += g.discriminator_loss();
            total_loss_generator += g.generator_loss();
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << " batch  = " << k << " sample = " << i / batch_size << std::endl;
        }
        generator->pre_epoch(k);
        if (k % 50 == 0/* && k > 0*/)
        {
            discriminator->serialize(dis_ser);
            generator->serialize(gen_ser);
            s1->save();
            s2->save();
        }
        l_plot->add_point(total_loss_disc, "discrminator", "r");
        l_plot->add_point(total_loss_generator, "generator", "b");
    }
}

void cat_gan2(shape_plot* sh_plot, loss_plot* l_plot)
{
    std::vector<std::string> cat_image_paths;
    std::map<int, std::string> paths =
    {
        { 1707, "D:\\Cats\\archive\\1707\\" },
        //{ 3325, "D:\\Cats\\archive\\3325\\" },
        //{ 5082, "D:\\Cats\\archive\\5082\\" },
        //{ 5893, "D:\\Cats\\archive\\5893\\" },
        //{ 7289, "D:\\Cats\\archive\\7289\\" },
        //{ 8610, "D:\\Cats\\archive\\8610\\" },
        //{ 9998, "D:\\Cats\\archive\\9998\\" }
    };
    int count = 0;
    for (auto it = paths.begin(); it != paths.end(); it++)
    {
        char buff[20];
        for (; count < it->first; count++)
        {
            std::string filename = it->second + std::string(itoa(count, buff, 10)) + ".png";
            cat_image_paths.push_back(filename);
        }
    }

    // params  of training
    size_t max_epochs = 50;
    size_t batch_size = 10;
    shape input_shape(256, 256, 3, 10);

    std::vector<float> batch_data;
    batch_data.resize(input_shape.size());
    std::vector<float> swap_buffer;
    std::vector<float> whole_data_v;
    whole_data_v.resize(input_shape.volume() * cat_image_paths.size());

    auto whole_loader = [&](size_t index) {
        swap_buffer.resize(input_shape.size());
        for (size_t i = index; i < index + batch_size; i++)
        {
            memcpy(batch_data.data() + (i - index) * input_shape.volume(), whole_data_v.data() + (i % cat_image_paths.size()) * input_shape.volume(), input_shape.volume() * sizeof(float));
        }
    };

    auto whole_data = [&]() {
        for (size_t i = 0; i < cat_image_paths.size(); i++)
        {
            auto img = load_image_normalized(cat_image_paths[i % cat_image_paths.size()].c_str(), input_shape.width, input_shape.height);
            memcpy(whole_data_v.data() + i * input_shape.volume(), img.pixels.data(), img.pixels.size() * sizeof(float));
        }
    };

    auto loader = [&](size_t index) {
        swap_buffer.resize(input_shape.size());
        size_t i = index;
        for (; i < index + batch_size; i++)
        {
            auto img = load_image_normalized(cat_image_paths[i % cat_image_paths.size()].c_str(), input_shape.width, input_shape.height);
            memcpy(swap_buffer.data() + (i - index) * input_shape.volume(), img.pixels.data(), img.pixels.size() * sizeof(float));
        }
    };

    model* encoder = new model(input_shape);

    filter_options opt;
    opt.w = 3;
    opt.h = 3;
    opt.zeropadding = true;
    opt.num_of_filters = 64;
    opt.stride = 1;
    encoder->addLayer(new conv2d_cudnn(opt));
    encoder->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));
    encoder->addLayer(new max_pool(2));
    opt.num_of_filters = 32;
    encoder->addLayer(new conv2d_cudnn(opt));
    encoder->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));
    encoder->addLayer(new max_pool(2));
    opt.num_of_filters = 16;
    encoder->addLayer(new conv2d_cudnn(opt));
    encoder->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));
    encoder->addLayer(new max_pool(2));
    opt.num_of_filters = 16;
    encoder->addLayer(new conv2d_cudnn(opt));
    encoder->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));
    encoder->addLayer(new conv2d_transposed(16, 2, 2, conv2d_transposed::padding::SAME));
    opt.num_of_filters = 32;
    encoder->addLayer(new conv2d_cudnn(opt));
    encoder->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));
    encoder->addLayer(new conv2d_transposed(32, 2, 2, conv2d_transposed::padding::SAME));
    opt.num_of_filters = 64;
    encoder->addLayer(new conv2d_cudnn(opt));
    encoder->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));
    encoder->addLayer(new conv2d_transposed(64, 2, 2, conv2d_transposed::padding::SAME));
    opt.num_of_filters = 3;
    encoder->addLayer(new conv2d_cudnn(opt));
    encoder->addLayer(new activation_layer(activation_layer::activation_function::tanh));
    encoder->set_learning_rate(0.01f);
    encoder->set_optimizer<momentum_optimizer>(0.9f);
    //std::thread tr(loader, 0);
    //tr.join();
    whole_data();

    // training loop
    for (size_t e = 0; e < max_epochs; e++)
    {
        for (size_t b = 0; b < cat_image_paths.size(); b += batch_size)
        {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            // calc
            //std::swap(swap_buffer, batch_data);
            //tr = std::thread(loader, b);
            whole_loader(b);
            encoder->getInputLayer().set_input(batch_data);
            encoder->predict();
            encoder->loss_layer().set_observed(batch_data);
            encoder->backprop();
            std::vector<float> image = (*encoder)[19]->get_native_output();
            image.resize(256 * 256 * 3);
            sh_plot->set_scale(4.0f);
            sh_plot->draw_rgb_channels(std::move(image), 32, 32, -1.0f, 1.0f);
            //tr.join();
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << " batch  = " << b / batch_size << " sample = " << b << std::endl;
        }
    }
}

void cifar(shape_plot* sh_plot, loss_plot* l_plot)
{
    std::vector<std_stream> cifar10;
    std::map<int, std::string> paths =
    {
        { 1, "C:\\Users\\wanks7a\\Downloads\\cifar-10-binary.tar\\cifar-10-binary\\cifar-10-batches-bin\\data_batch_1.bin" },
        //{ 2, "C:\\Users\\wanks7a\\Downloads\\cifar-10-binary.tar\\cifar-10-binary\\cifar-10-batches-bin\\data_batch_2.bin" },
        //{ 3, "C:\\Users\\wanks7a\\Downloads\\cifar-10-binary.tar\\cifar-10-binary\\cifar-10-batches-bin\\data_batch_3.bin" },
        //{ 4, "C:\\Users\\wanks7a\\Downloads\\cifar-10-binary.tar\\cifar-10-binary\\cifar-10-batches-bin\\data_batch_4.bin" },
        //{ 5, "C:\\Users\\wanks7a\\Downloads\\cifar-10-binary.tar\\cifar-10-binary\\cifar-10-batches-bin\\data_batch_5.bin" },
        //{ 6, "C:\\Users\\wanks7a\\Downloads\\cifar-10-binary.tar\\cifar-10-binary\\cifar-10-batches-bin\\test_batch.bin" },
    };
    
    for (auto it = paths.begin(); it != paths.end(); it++)
    {
        cifar10.push_back(std_stream(it->second));
    }
    
    // params  of training
    size_t max_epochs = 10000;
    size_t batch_size = 256;
    shape input_shape(32, 32, 3, batch_size);
    const int examples = 60000;
    std::vector<float> batch_data;
    batch_data.resize(input_shape.size());
    std::vector<float> swap_buffer;
    std::vector<float> whole_data_v;

    whole_data_v.resize(input_shape.volume() * examples);

    for (size_t i = 0; i < cifar10.size(); i++)
    {
        cifar10[i].read();
        std::vector<float> data;

        int cifar_pixels = 0;
        for (size_t j = 1; j < cifar10[i].data.size(); j++)
        {   
            if (cifar_pixels != 3072)
            {
                data.push_back((static_cast<unsigned char>(cifar10[i].data[j]) - 127.5f) / 127.5f);
                cifar_pixels++;
            }
            else
                cifar_pixels = 0;
        }

        memcpy(whole_data_v.data() + (i * 10000) * input_shape.volume(), data.data(), data.size() * sizeof(float));
    }

    auto whole_loader = [&](size_t index, std::vector<float>& d, shape inp_shape, const std::vector<float>& whole_D) {
        for (size_t i = index; i < index + batch_size; i++)
        {
            memcpy(d.data() + (i - index) * inp_shape.volume(), whole_D.data() + (i % examples) * inp_shape.volume(), inp_shape.volume() * sizeof(float));
        }
    };

    model* encoder = new model(input_shape);

    filter_options opt;
    opt.w = 3;
    opt.h = 3;
    opt.zeropadding = true;
    opt.num_of_filters = 64;
    opt.stride = 1;
    // encoder
    encoder->addLayer(new conv2d_cudnn(opt));
    encoder->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));
    opt.num_of_filters = 128;
    opt.stride = 3;
    encoder->addLayer(new conv2d_cudnn(opt));
    encoder->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));
    opt.num_of_filters = 128;
    encoder->addLayer(new conv2d_cudnn(opt));
    encoder->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));
    opt.num_of_filters = 256;
    encoder->addLayer(new conv2d_cudnn(opt));
    encoder->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));
    encoder->addLayer(new dense_gpu(1));
    encoder->addLayer(new activation_layer(activation_layer::activation_function::Sigmoid));

    // decoder
    shape decoder_inpu_shape(100, 1, 1, batch_size);
    model* decoder = new model(decoder_inpu_shape);
    decoder->addLayer(new dense_gpu(4 * 4 * 5, false));
    decoder->addLayer(new reshape_layer(shape(4, 4, 5)));
    decoder->addLayer(new batch_norm_cuda());
    decoder->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));

    decoder->addLayer(new conv2d_transposed(512, 4, 2, conv2d_transposed::padding::SAME));
    decoder->addLayer(new batch_norm_cuda());
    decoder->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));

    decoder->addLayer(new conv2d_transposed(256, 4, 2, conv2d_transposed::padding::SAME));
    decoder->addLayer(new batch_norm_cuda());
    decoder->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));

    decoder->addLayer(new conv2d_transposed(128, 4, 2, conv2d_transposed::padding::SAME));
    decoder->addLayer(new batch_norm_cuda());
    decoder->addLayer(new activation_layer(activation_layer::activation_function::LeakyReLU));

    opt.num_of_filters = 3;
    opt.w = 3;
    opt.h = 3;
    opt.stride = 1;
    decoder->addLayer(new conv2d_cudnn(opt));
    decoder->addLayer(new activation_layer(activation_layer::activation_function::tanh));
    gan cifar;
    cifar.set_discriminator(encoder);
    cifar.set_generator(decoder);

    std::vector<float> fake_data;
    std::vector<float> fake_data_batch;
    
    auto noise = [&]() {
        size_t currentSize = examples * decoder_inpu_shape.volume();
        fake_data.reserve(currentSize);
        std::mt19937_64 rng;
        // initialize the random number generator with time-dependent seed
        uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::seed_seq ss{ uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32) };
        rng.seed(ss);

        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

        for (int i = 0; i < currentSize; i++)
        {
            fake_data.emplace_back(distribution(rng));
        }
    };

    noise();
    fake_data_batch.resize(decoder_inpu_shape.size());

    auto s1 = std::make_shared<std_stream>("disc_cats.b");
    auto s2 = std::make_shared<std_stream>("gen_cats.b");
    auto s3 = std::make_shared<std_stream>("gen_cats.b");

    binary_serialization dis_ser(s1);
    binary_serialization gen_ser(s2);
    //binary_serialization gen_ser3(s3);
    //s3->read();
    //auto m = gen_ser3.deserialize_model();
    s1->read();
    s2->read();
    dis_ser.deserialize_model(*encoder);
    gen_ser.deserialize_model(*decoder);
    encoder->set_optimizer<momentum_optimizer>(0.1f);
    decoder->set_optimizer<momentum_optimizer>(0.1f);
    encoder->set_learning_rate(0.01f);
    decoder->set_learning_rate(0.01f);


    // training loop
    for (size_t e = 0; e < max_epochs; e++)
    {
        double total_loss_generator = 0.0;
        double total_loss_disc = 0.0;
        for (size_t b = 0; b < examples; b += batch_size)
        {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            // calc
            //std::swap(swap_buffer, batch_data);
            //tr = std::thread(loader, b);
            whole_loader(b, batch_data, input_shape, whole_data_v);
            whole_loader(b, fake_data_batch, decoder_inpu_shape, fake_data);

            cifar.predict(batch_data, fake_data_batch);
            cifar.backprop();


            std::vector<float> image = (*decoder)[14]->get_native_output();
            image.resize(32 * 32 * 3);
            sh_plot->set_scale(4.0f);
            sh_plot->draw_rgb_channels(std::move(image), 32, 32, -1.0f, 1.0f);
            total_loss_disc += cifar.discriminator_loss() / batch_size;
            total_loss_generator += cifar.generator_loss() / batch_size;
            //tr.join();
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << " batch  = " << b / batch_size << " sample = " << b << std::endl;
        }
        if (e % 25 == 0)
        {
            encoder->serialize(dis_ser);
            decoder->serialize(gen_ser);
            s1->save();
            s2->save();
        }
        decoder->pre_epoch(e);
        l_plot->add_point(total_loss_disc, "discrminator", "r");
        l_plot->add_point(total_loss_generator, "generator", "b");
    }
}

int main(int argc, char* argv[])
{
    viewer view_manager;
    shape_plot* sh_plot = new shape_plot(1024, 768, "shape");
    loss_plot* l_plot = new loss_plot(1024, 768, "metrics");
    view_manager.add_view(sh_plot);
    view_manager.add_view(l_plot);
    std::thread t([&]() {
        startTrain(sh_plot, l_plot);
        //gan_test(sh_plot, l_plot);
        //cifar(sh_plot, l_plot);
        //cat_gan2(sh_plot, l_plot);
        });
    view_manager.loop();
	return 0;
}