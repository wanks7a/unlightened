#include "python_api.h"
#include <binary_serialization.h>
#include <read_write_stream.h>
#include <NeuralNet.h>
#include <iostream>

static std::shared_ptr<model> load_model(const char* path)
{
    std::shared_ptr<generic_stream> stream(new read_write_stream(path));
    binary_serialization ser(stream);
    return ser.deserialize_model();
}

extern "C"
{
    __declspec(dllexport) bool predict(const char* path, const float* input, size_t size, float* out, size_t out_size)
    {
        auto model = load_model(path);
        if (model->getInputLayer().get_shape().size() != size || 
            model->loss_layer().get_shape().size() != out_size)
            return false;
        model->getInputLayer().set_input(input, size);
        model->predict();
        return true;
    }

    __declspec(dllexport) void print_model(const char* path)
    {

    }

    __declspec(dllexport) void print_input_shape(const char* path)
    {
        auto current_model = load_model(path);
        auto shape = current_model->getInputLayer().get_shape();
        std::cout << "width: " << shape.width << " height: " << shape.height << " depth: " << shape.depth << std::endl;
    }

    __declspec(dllexport) void print_output_shape(const char* path)
    {
        auto current_model = load_model(path);
        auto shape = current_model->loss_layer().get_shape();
        std::cout << "width: " << shape.width << " height: " << shape.height << " depth: " << shape.depth << std::endl;
    }

    __declspec(dllexport) void get_input_shape(const char* path)
    {

    }

    __declspec(dllexport) void get_output_shape(const char* path)
    {

    }
}