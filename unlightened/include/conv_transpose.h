#pragma once
#include <cudnn_helpers.h>
#include <conv_filter.h>
#include <serializable_interface.h>


class conv2d_transposed : public serializable_layer<conv2d_transposed>
{
public:
    enum class padding
    {
        VALID,
        SAME,
        FULL
    };
private:
    cuVector<float> shared_mem;
    padding pad_type;
    filter filter_data;
    cudnn_descriptor4d input_desc;
    cudnn_descriptor4d out_desc;
    cudnn_descriptor4d bias_desc;
    cudnn_descriptor4d weight_desc;
    cudnn_filter_descriptor filter_desc;
    cudnn_conv2d_descriptor conv2d_desc;
    cudnn_add_tensor        add_tensor;
    cudnn_hanlde            handle;

    cudnnConvolutionFwdAlgo_t conv2d_forwardpass_alg;
    cudnnConvolutionBwdFilterAlgo_t filter_backprop_algo;
    cudnnConvolutionBwdDataAlgo_t backprop_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

    cuVector<float> output;
    cuVector<float> deriv_output;
    const float* input_from_prev_layer = nullptr;
    cuVector<float> layer_input;
public:
    conv2d_transposed();
    conv2d_transposed(size_t number_filters, size_t filter_size, size_t stride, padding pad);
    void init(const shape& input) override;
    void forward_pass(Layer* prevLayer) override;
    void backprop(Layer* layer) override;
    const float* get_output() const override;
    const float* derivative_wr_to_input() const override;
    size_t calc_output_dim(size_t input) const;

    template <typename Serializer>
    void serialize_members(Serializer& s) const
    {
        s << static_cast<char>(pad_type) << filter_data.get_options() << filter_data.get_weights() << filter_data.get_bias();
    }

    template <typename Serializer>
    void deserialize_members(Serializer& s)
    {
        char pad_temp;
        filter_options opt;
        s >> pad_temp >> opt;
        pad_type = static_cast<padding>(pad_temp);
        filter_data.set_options(opt);
        init(input_shape);
        s >> filter_data.get_weights() >> filter_data.get_bias();
    }

    filter& get_filter()
    {
        return filter_data;
    }
private:
    void init_tensors();
    bool cudnn_status_check(cudnnStatus_t status) const;
    void backprop_cudnn(const float* derivative);
    void backprop_weights_cudnn(const float* derivative, const float* input);
    void update_weights();
};