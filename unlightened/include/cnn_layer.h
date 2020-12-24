#pragma once
#include <Layer.h>
#include <conv_filter.h>
#include <cudnn.h>

void backprop_weights_3d(const float* input, const shape& input_shape, float* output, const shape& output_shape, const float* weights, unsigned int filter_row, unsigned int filter_col, unsigned int filter_offset, unsigned int weights_offset_batch);
void conv_3d(const float* input, const shape& input_shape, float* output, const shape& output_shape, const float* weights, const float* bias, unsigned int filter_row, unsigned int filter_col, unsigned int offset);
void merge_conv_with_bias(const float* input, const shape& input_shape, const float* bias_vector, float* output, const unsigned int batch_offset);
void flip_filter(float* input, const shape& filter_shape, bool horizontal_lines);
void derivative_input_3d(const float* input, const shape& input_shape, float* output, const shape& output_shape, const float* weights,
    unsigned int filter_row,
    unsigned int filter_col,
    unsigned int filter_offset,
    unsigned int weights_offset_batch);
void update_weights(const float* weights_error, shape weights_shape, unsigned int num_of_filters, float* weights, float learning_rate);
void update_bias(const float* derivative, shape derivative_shape, float* bias, float learning_rate);
void add_bias_to_output(cuVector<float>& output, const shape& output_shape, cuVector<float>& bias);

class cnn_layer : public Layer
{
    filter_options options;
    filter filters;
    size_t filters_size;
    cuVector<float> output;
    cuVector<float> input_derivative;
    Layer* input_layer;
    cuVector<float> layer_input;
    const float* input = nullptr;
    bool is_first_layer;
    bool use_cudnn = false;
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnConvolutionDescriptor_t convolution_forwardpass_descriptor;
    cudnnConvolutionFwdAlgo_t convolution_forwardpass_algorithm;
    cudnnConvolutionBwdDataAlgo_t backprop_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    cuVector<float> cudnn_memory_forward_pass;
    cuVector<float> cudnn_memory_backprop;
public:
    cnn_layer(size_t filter_dimension, size_t num_of_filters, bool first_layer = false);
    void init(const shape& input) override;
    void forward_pass(Layer* prevLayer) override;
    void backprop(Layer* layer) override;
    const float* get_output() const override;
    const float* derivative_wr_to_input() const override;

    filter_options get_options() const
    {
        return options;
    }

    filter& get_filters()
    {
        return filters;
    }

    void set_options(const filter_options& opt)
    {
        options = opt;
    }

    cuVector<float>& get_bias_vector()
    {
        return filters.get_bias();
    }

    ~cnn_layer();

private:
    void init_cudnn();
    void checkCUDNN(const cudnnStatus_t& status);
    void backprop_cudnn(const float* derivative);
};