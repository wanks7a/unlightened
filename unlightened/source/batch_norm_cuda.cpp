#include <batch_norm_cuda.h>
#include <algorithm>

void batch_norm_cuda::init(const shape& input)
{
    output_shape = input;
    input_desc.create(input.width, input.height, input.depth, input.batches);
    output_desc.create(input.width, input.height, input.depth, input.batches);
    batch_norm_desc.create(1, 1, input.depth, 1);
    CUDA_CHECK(cudnnDeriveBNTensorDescriptor(batch_norm_desc.descriptor, input_desc.descriptor, CUDNN_BATCHNORM_SPATIAL));
    running_mean_data.resize(input.depth, 0.0f);
    running_variance_data.resize(input.depth, 0.0f);
    save_mean_data.resize(input.depth, 0.0f);;
    save_inv_variance_data.resize(input.depth, 0.0f);;
    output.reserve(input.size());
    scale_data.resize(input.depth, 1.0f);
    scale_data_derivative.resize(input.depth, 0.0f);
    bias_data.resize(input.depth, 0.0f);
    bias_data_derivative.resize(input.depth, 0.0f);
    input_derivative.resize(input.size(), 0.0f);
}


const float* batch_norm_cuda::get_output() const
{
    return output.data();
}

const float* batch_norm_cuda::derivative_wr_to_input() const
{
    return input_derivative.data();
}


void batch_norm_cuda::device_forward()
{
    float alpha = 1.0f, beta = 0.0f;
    const double epsilon = std::max(bn_eps, CUDNN_BN_MIN_EPSILON);
    CUDA_CHECK(cudnnBatchNormalizationForwardTraining(handle.handle,
        CUDNN_BATCHNORM_SPATIAL,
        &alpha,
        &beta,
        input_desc.descriptor, get_forward_input(),
        output_desc.descriptor, output.data(),
        batch_norm_desc.descriptor, scale_data.data(), bias_data.data(),
        momentum,
        running_mean_data.data(), running_variance_data.data(),
        epsilon,
        save_mean_data.data(), save_inv_variance_data.data()));
    factor++;
    momentum = 1.0 / (1.0 + factor);
}

void batch_norm_cuda::device_backprop()
{
    float alpha = 1.0f, beta = 0.0f;
    const double epsilon = std::max(bn_eps, CUDNN_BN_MIN_EPSILON);
    CUDA_CHECK(cudnnBatchNormalizationBackward(handle.handle,
        CUDNN_BATCHNORM_SPATIAL,
        &alpha,
        &beta,
        &alpha,
        &alpha,
        input_desc.descriptor, get_forward_input(),
        output_desc.descriptor, get_backprop_input(),
        input_desc.descriptor, input_derivative.data(),
        batch_norm_desc.descriptor, scale_data.data(),
        scale_data_derivative.data(), bias_data_derivative.data(),
        epsilon,
        save_mean_data.data(), save_inv_variance_data.data()));
}

weights_properties batch_norm_cuda::get_weights() const
{
    weights_properties props;
    props.size = scale_data.size();
    props.ptr = scale_data.data();
    return props;
};

weights_properties batch_norm_cuda::get_weights_deriv() const
{
    weights_properties props;
    props.size = scale_data_derivative.size();
    props.ptr = scale_data_derivative.data();
    return props;
};

weights_properties batch_norm_cuda::get_bias() const
{
    weights_properties props;
    props.size = bias_data.size();
    props.ptr = bias_data.data();
    return props;
};

weights_properties batch_norm_cuda::get_bias_deriv() const
{
    weights_properties props;
    props.size = bias_data_derivative.size();
    props.ptr = bias_data_derivative.data();
    return props;
};

void batch_norm_cuda::pre_epoch(size_t epoch)
{
    factor = 1;
    momentum = 1;
}

void batch_norm_cuda::post_epoch(size_t epoch)
{
}