#pragma once
#include <batch_norm.h>
#include <cuda_device.h>
#include <cudnn_helpers.h>

class batch_norm_cuda : public batch_norm<cuda_device>
{
    cudnn_hanlde handle;
    cudnn_descriptor4d input_desc;
    cudnn_descriptor4d output_desc;
    cudnn_descriptor4d batch_norm_mean_desc;
    cudnn_descriptor4d batch_norm_variance_desc;
    cudnn_descriptor4d batch_norm_desc;
    device_vector<cuda_device, float> running_mean_data;
    device_vector<cuda_device, float> running_variance_data;
    device_vector<cuda_device, float> save_mean_data;
    device_vector<cuda_device, float> save_inv_variance_data;
    device_vector<cuda_device, float> output;
    device_vector<cuda_device, float> scale_data;
    device_vector<cuda_device, float> bias_data;
    device_vector<cuda_device, float> input_derivative;
    device_vector<cuda_device, float> scale_data_derivative;
    device_vector<cuda_device, float> bias_data_derivative;
    double bn_eps = 0.0001;
    size_t factor = 0;
    double momentum = 0.0;
public:
    void device_forward() override;
    void device_backprop() override;

    void init(const shape& input) override;

    const float* get_output() const override;

    const float* derivative_wr_to_input() const override;

    weights_properties get_weights() const override;

    weights_properties get_weights_deriv() const override;

    weights_properties get_bias() const override;

    weights_properties get_bias_deriv() const override;

    void pre_epoch(size_t epoch) override;

    void post_epoch(size_t epoch) override;


    batch_norm_cuda() {};
};