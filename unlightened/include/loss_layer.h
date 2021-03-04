#pragma once
#include <serializable_interface.h>
#include <device_vector.h>
#include <cuda_device.h>

enum class LOSS
{
    MSE = 1,
    binary_cross_entropy = 2
};

class loss_layer_cpu : public serializable_layer<loss_layer_cpu>
{
    LOSS loss_type;
    std::vector<float> predictedValue;
    std::vector<float> derivativeWRToInput;
    std::vector<float> observedValues;
    mutable device_vector<cuda_device, float> derivative_device;
    mutable device_vector<cuda_device, float> predicted_device;
    size_t size;
public:

    loss_layer_cpu() : size(0), loss_type(LOSS::MSE)
    {
        in_device_memory = true;
    }

    void init(const shape& input) override;

    void forward_pass(Layer* prevLayer) override;

    void backprop_mse();

    void backprop_binary_cross_ent();

    void backprop(Layer* layer) override;

    bool set_observed(const std::vector<float>& observedVal);

    bool set_observed(const float* ptr, size_t size);

    bool set_derivative_manual(const std::vector<float>& deriv);

    bool set_derivative_manual(const float* deriv, size_t size);

    void set_loss_func(LOSS loss)
    {
        loss_type = loss;
    }

    const float* get_output() const override;

    const float* derivative_wr_to_input() const override;

    void print_predicted(size_t examples) const;

    double get_binary_cross_ent_mean_loss() const;

    double get_mse_mean() const;

    float get_mean_loss() const;

    template <typename Serializer>
    void serialize_members(Serializer& s) const
    {
        char type = static_cast<char>(loss_type);
        s << type;
    }

    template <typename Serializer>
    void deserialize_members(Serializer& s)
    {
        init(input_shape);
        char type;
        s >> type;
        switch (type)
        {
            case LOSS::MSE: loss_type = LOSS::MSE; break;
            case LOSS::binary_cross_entropy: loss_type = LOSS::binary_cross_entropy; break;
            default: loss_type = LOSS::MSE; break;
        }
    }

    ~loss_layer_cpu() = default;
};