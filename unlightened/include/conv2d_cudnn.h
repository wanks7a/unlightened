#pragma once
#include <conv_filter.h>
#include <cudnn.h>
#include <device_layer.h>
#include <cuda_device.h>

class conv2d_cudnn : public device_layer_new<conv2d_cudnn, cuda_device>
{
protected:
    bool initialized = false;
    filter_options options;
    filter filters;
    size_t filters_size;
    cuVector<float> output;
    cuVector<float> input_derivative;
    Layer* input_layer;
    cuVector<float> layer_input;
    const float* input = nullptr;
    bool is_first_layer;
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t input_descriptor = nullptr;
    cudnnTensorDescriptor_t output_descriptor = nullptr;
    cudnnFilterDescriptor_t filter_descriptor = nullptr;
    cudnnConvolutionDescriptor_t convolution_forwardpass_descriptor = nullptr;
    cudnnConvolutionFwdAlgo_t convolution_forwardpass_algorithm;
    cudnnConvolutionBwdDataAlgo_t backprop_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    cudnnConvolutionBwdFilterAlgo_t filter_backprop_algo;
    cudnnOpTensorDescriptor_t add_op_descriptor = nullptr;
    cudnnTensorDescriptor_t bias_tensor_descriptor = nullptr;
    cudnnTensorDescriptor_t weights_tensor_descriptor = nullptr;
    cuVector<float> cudnn_memory_forward_pass;
    cuVector<float> cudnn_memory_backprop;
    cuVector<float> cudnn_memory_backprop_filter;
public:
    conv2d_cudnn();
    conv2d_cudnn(size_t filter_dimension, size_t num_of_filters, bool first_layer = false);
    conv2d_cudnn(const filter_options& opt, bool first_layer = false);
    void init(const shape& input) override;
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

    template <typename Serializer>
    void serialize_members(Serializer& s) const
    {
        s << is_first_layer << filters.get_options() << filters.get_weights() << filters.get_bias();
    }

    template <typename Serializer>
    void deserialize_members(Serializer& s)
    {
        s >> is_first_layer >> options;
        filters_size = options.num_of_filters;
        init(input_shape);
        s >> filters.get_weights() >> filters.get_bias();
    }

    weights_properties get_weights() const override;

    weights_properties get_weights_deriv() const override;

    weights_properties get_bias() const override;

    weights_properties get_bias_deriv() const override;

    bool forward(std::shared_ptr<cuda_device>& d, const blob_view<float>& input_data);
    bool backward(std::shared_ptr<cuda_device>& d, const blob_view<float>& derivative_data);

    virtual ~conv2d_cudnn();

protected:
    void init_cudnn();
    void checkCUDNN(const cudnnStatus_t& status);
    void backprop_cudnn(const float* derivative);
    void backprop_weights_cudnn(const float* derivative);
public:
    void update_weights();
};