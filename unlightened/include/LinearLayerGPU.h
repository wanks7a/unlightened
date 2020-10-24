#pragma once
#include <Layer.h>
#include <vector>
#include <memory>
#include <device_memory.h>
#include <math.h>

void linearLayerForwardPassGPU(float* output, const float* weights, const float* input, const shape& input_shape, const float* bias, const shape& output_shape);
void calcDerivativeWRtoInput(float* derivativeWRtoInput, size_t inputSize, const float* derivateWRtoOutput, shape output_shape, const float* weights);
void updateWeights(float* weights, const float* derivativeWRtoOutput, const float* input, size_t input_size, size_t outputSize, shape out_shape, float learning_rate);
void updateBias(float* bias, const float* derivative_wr_to_out, size_t output_size, shape output_shape, float learning_rate);

class dense_gpu : public Layer
{
private:
    std::vector<float> weight;
    cuVector<float> weightsGPU;
    cuVector<float> biasGPU;
    cuVector<float> outputGPU;
    cuVector<float> derivativeWRtoInputGPU;
    cuVector<float> inputVectorGPU;
    const float* inputPtr;
    size_t size;
    size_t input_size;
    size_t input_size_with_bias;
public:

    dense_gpu(size_t neuron_size) : size(neuron_size)
    {
        device_layer = true;
    }

    void init(const shape& input) override
    {
        input_size = input.volume();
        weight.resize(input_size * size);
        derivativeWRtoInputGPU.resize(input.size(), 0.0f);
        biasGPU.resize(size, 0.0f);
        outputGPU.resize(size * input.batches, 0.0f);
        weightsGPU.setValues(weight);
        weightsGPU.randomize();
        float fan_in = static_cast<float>(input_size);
        weightsGPU *= sqrtf(2.0f / fan_in);
     
        output_shape.width = size;
        output_shape.batches = input_shape.batches;
    }

    bool set_weights(const std::vector<float>& w)
    {
        if (weight.size() == w.size())
        {
            weight = w;
            return weightsGPU.setValues(weight);
        }
        return false;
    }

    void forward_pass(Layer* prevLayer) override
    {
        inputPtr = prevLayer->get_output();
        shape out_shape = output_shape;

        shape input_shape;
        input_shape.width = prevLayer->get_shape().volume(); // represent the value as 1d array
        input_shape.batches = prevLayer->get_shape().batches;
        auto v = prevLayer->get_native_output();
        if (!prevLayer->is_device_layer())
        {
            inputVectorGPU = prevLayer->get_device_output();
            inputPtr = inputVectorGPU.get();
            linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr, input_shape, biasGPU.get(), out_shape);
        }
        else
        {
            linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr, input_shape, biasGPU.get(), out_shape);
        }
    }

    void backprop(Layer* layer) override
    {
        auto v = layer->get_native_derivative();
        shape temp_out_shape = output_shape;
        if (layer->is_device_layer())
        {
            calcDerivativeWRtoInput(derivativeWRtoInputGPU.get(), input_size, layer->derivative_wr_to_input(), temp_out_shape, weightsGPU.get());
            if (update_on_backprop)
            {
                updateWeights(weightsGPU.get(), layer->derivative_wr_to_input(), inputPtr, input_size, size, output_shape, learing_rate);
                updateBias(biasGPU.get(), layer->derivative_wr_to_input(), size, output_shape, learing_rate);
            }
        }
        else
        {
            cuVector<float> derivativeWRToOutput = layer->get_device_derivative();
            calcDerivativeWRtoInput(derivativeWRtoInputGPU.get(), input_size, derivativeWRToOutput.get(), temp_out_shape, weightsGPU.get());
            if (update_on_backprop)
            {
                updateWeights(weightsGPU.get(), derivativeWRToOutput.get(), inputPtr, input_size, size, output_shape, learing_rate);
                updateBias(biasGPU.get(), derivativeWRToOutput.get(), size, output_shape, learing_rate);
            }
        }
    }

    const float* get_output()
    {
        return outputGPU.get();
    };
    const float* derivative_wr_to_input()
    {
        return derivativeWRtoInputGPU.get();
    }

    ~dense_gpu()
    {
    }
};