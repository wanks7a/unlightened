#include <loss_layer.h>
#include <math.h>

void loss_layer_cpu::init(const shape& input)
{
    output_shape = input;
    size = input.size();
    derivativeWRToInput.resize(size);
    observedValues.resize(size);
}

void loss_layer_cpu::forward_pass(Layer* prevLayer)
{
    predictedValue = prevLayer->get_native_output();
};

void loss_layer_cpu::backprop_mse()
{
    for (size_t i = 0; i < size; i++)
    {
        derivativeWRToInput[i] = -2 * (observedValues[i] - predictedValue[i]);
    }
}

void loss_layer_cpu::backprop(Layer* layer)
{
    switch (loss_type)
    {
    case LOSS::MSE: backprop_mse();
        break;
    case LOSS::binary_cross_entropy: backprop_binary_cross_ent();
        break;
    }
}

bool loss_layer_cpu::set_observed(const std::vector<float>& observedVal)
{
    if (observedVal.size() == observedValues.size())
    {
        observedValues = observedVal;
        return true;
    }
    return false;
}

bool loss_layer_cpu::set_observed(const float* ptr, size_t size)
{
    if (size == observedValues.size())
    {
        memcpy(observedValues.data(), ptr, sizeof(float) * size);
        return true;
    }
    return false;
}

bool loss_layer_cpu::set_derivative_manual(const std::vector<float>& deriv)
{
    if (derivativeWRToInput.size() == deriv.size())
    {
        derivativeWRToInput = deriv;
        return true;
    }
    else
        std::cout << "Manual derivative size is wrong " << std::endl;
    return false;
}

bool loss_layer_cpu::set_derivative_manual(const float* ptr, size_t size)
{
    if (derivativeWRToInput.size() == size)
    {
        memcpy(derivativeWRToInput.data(), ptr, sizeof(float) * size);
        return true;
    }
    else
        std::cout << "Manual derivative size is wrong " << std::endl;
    return false;
}


double loss_layer_cpu::get_mse_mean() const
{
    float result = 0;
    for (size_t i = 0; i < size; i++)
    {
        result += ((observedValues[i] - predictedValue[i]) * (observedValues[i] - predictedValue[i])) / output_shape.batches;
    }
    return result;
}

float loss_layer_cpu::get_mean_loss() const
{
    float result = 0;
    switch (loss_type)
    {
    case LOSS::MSE: result = get_mse_mean();
        break;
    case LOSS::binary_cross_entropy: result = get_binary_cross_ent_mean_loss();
        break;
    }
    return result;
}

double loss_layer_cpu::get_binary_cross_ent_mean_loss() const
{
    double result = 0;
    for (size_t i = 0; i < size; i++)
    {
        result += (observedValues[i] * log(predictedValue[i]) + (1 - observedValues[i]) * log(1 - predictedValue[i])) / output_shape.batches;
    }
    return -result / output_shape.batches;
}

void loss_layer_cpu::backprop_binary_cross_ent()
{
    for (size_t i = 0; i < size; i++)
    {   
        derivativeWRToInput[i] = -((observedValues[i] / predictedValue[i]) - (1 - observedValues[i]) / (1 - predictedValue[i]));
    }
}

void loss_layer_cpu::print_predicted(size_t examples) const
{
    for (size_t i = 0; i < examples; i++)
    {
        std::cout << "Actual " << " Predicted " << std::endl;
        for (size_t j = 0; j < output_shape.volume(); j++)
        {
            printf("   %.5f      %.5f\n", observedValues[i * output_shape.volume() + j], predictedValue[i * output_shape.volume() + j]);
        }
    }
}