
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

int window_width = 1024;
int window_height = 768;

float findMax(const float* ptr, size_t size)
{
    float max = ptr[0];
    for (size_t i = 0; i < size; i++)
    {
        if (max < ptr[i])
            max = ptr[i];
    }
    return max;
}

Uint32 leprGrayScale(float input, float max)
{
    if (input < 0)
        input = 0;
    
    Uint32 value = (input / max) * 255;
    Uint32 result = 0;
    result |= value;
    result |= (value << 8);
    result |= (value << 16);
    return result;
}

void updateWindow(SDL_Window& window, SDL_Renderer& renderer, shape& input_shape, std::vector<float>& data)
{
    std::vector<SDL_Surface*> surfaces;
    std::vector<SDL_Texture*> images;
    std::vector<SDL_Rect> screen_dest;

    std::vector<Uint32> single_pic(input_shape.area());
    Uint32 rmask, gmask, bmask, amask;
    rmask = 0x000000ff;
    gmask = 0x0000ff00;
    bmask = 0x00ff0000;
    amask = 0;
    int startW = 0, startH = 0;
    for (size_t i = 0; i < input_shape.depth; i++)
    {
        float max = findMax(&data[i * input_shape.area()], input_shape.area());
        for (size_t j = 0; j < input_shape.area(); j++)
        {
            single_pic[j] = leprGrayScale(data[i * input_shape.area() + j], max);
        }
        SDL_Surface* surf = SDL_CreateRGBSurfaceFrom((void*)single_pic.data(), input_shape.width, input_shape.height, 32, 4 * input_shape.width,
            rmask, gmask, bmask, amask);
        SDL_Texture* texture = SDL_CreateTextureFromSurface(&renderer, surf);
        images.push_back(texture);
        surfaces.push_back(surf);
        SDL_Rect dstrect = { startW, startH, input_shape.width*4, input_shape.height*4};
        screen_dest.push_back(dstrect);
        if ((startW + input_shape.width* 4) >= window_width)
        {
            startW = 0;
            startH += input_shape.height * 4;
        }
        else
        {
            startW += input_shape.width * 4;
        }
    }
    for (size_t i = 0; i < images.size(); i++)
    {
        SDL_RenderCopy(&renderer, images[i], NULL, &screen_dest[i]);
    }
    SDL_RenderPresent(&renderer);
    for (size_t i = 0; i < images.size(); i++)
    {
        SDL_DestroyTexture(images[i]);
        SDL_FreeSurface(surfaces[i]);
    }

}

void startTrain(SDL_Window& window, SDL_Renderer& renderer)
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
            updateWindow(window, renderer, shape, l);
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

int sample1(mglGraph* gr)
{
    gr->NewFrame();             // the first frame
    gr->Rotate(60, 40);
    gr->Box();
    gr->EndFrame();             // end of the first frame
    gr->NewFrame();             // the second frame
    gr->Box();
    gr->Axis("xy");
    gr->EndFrame();             // end of the second frame
    return 1;       // returns the frame number
}

int main(int argc, char* argv[])
{
    //mglGraph gr;
    //gr.Alpha(true);   gr.Light(true);
    //sample1(&gr);              // The same drawing function.
    //gr.WritePNG("test.png");  // Don't forget to save the result!
    

    SDL_Window* window;                    // Declare a pointer

    SDL_Init(SDL_INIT_VIDEO);              // Initialize SDL2

    // Create an application window with the following settings:
    window = SDL_CreateWindow(
        "An SDL2 window",                  // window title
        SDL_WINDOWPOS_UNDEFINED,           // initial x position
        SDL_WINDOWPOS_UNDEFINED,           // initial y position
        window_width,                               // width, in pixels
        window_height,                               // height, in pixels
        SDL_WINDOW_OPENGL                  // flags - see below
    );

    // Check that the window was successfully created
    if (window == NULL) {
        // In the case that the window could not be made...
        printf("Could not create window: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);
    // The window is open: could enter program loop here (see SDL_PollEvent())
    startTrain(*window, *renderer);

    SDL_Delay(3000);  // Pause execution for 3000 milliseconds, for example

    // Close and destroy the window
    SDL_DestroyWindow(window);

    // Clean up
    SDL_Quit();

	return 0;
}