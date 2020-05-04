#pragma once
#include <Layer.h>
#include <vector>
#include <memory>
#include <device_memory.h>
#include <math.h>

void linearLayerForwardPassGPU(float* output, const float* weights, const float* input, const shape& input_shape, const shape& output_shape, bool bias_subtracted);
void calcDerivativeWRtoInput(float* derivativeWRtoInput, size_t input_size, const float* derivateWRtoOutput, shape output_shape, const float* weights, bool last_out_is_bias);
void updateWeightsAndBias(float* weights, const float* derivativeWRtoOutput, const float* input, size_t input_size, size_t outputSize, shape out_shape, float learning_rate);

class LinearLayerGPU : public Layer
{
private:
    std::vector<float> weight;
    cuVector<float> weightsGPU;
    cuVector<float> outputGPU;
    cuVector<float> derivativeWRtoInputGPU;
    cuVector<float> inputVectorGPU;
    const float* inputPtr;
    bool add_bias_to_output;
    size_t size;
    size_t input_size;
public:

    LinearLayerGPU(size_t neuron_size, bool add_bias = true) : size(neuron_size), add_bias_to_output(add_bias)
    {
        device_layer = true;
    }

    void init(const shape& input) override
    {
        input_size = input.volume();
        weight.resize(input_size * size);
        derivativeWRtoInputGPU.resize(input.size(), 0.0f);
        // +1 is for the bias
        if(add_bias_to_output)
            outputGPU.resize((size + 1) * input.batches, 1.0f);
        else
            outputGPU.resize(size * input.batches, 1.0f);

        weightsGPU.setValues(weight);
        weightsGPU.randomize();
        float fan_in = static_cast<float>(input_size);
        weightsGPU *= sqrtf(2.0f / fan_in);
        
        if(add_bias_to_output)
            size = size + 1;

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
        if (add_bias_to_output)
            out_shape.width = out_shape.width - 1; // -1 because we dont want to calculate for the bias

        shape input_shape;
        input_shape.width = prevLayer->get_shape().volume(); // represent the value as 1d array
        input_shape.batches = prevLayer->get_shape().batches;
        
        if (!prevLayer->is_device_layer())
        {
            inputVectorGPU = prevLayer->get_device_output();
            inputPtr = inputVectorGPU.get();
            linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr, input_shape, out_shape, add_bias_to_output);
        }
        else
        {
            linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr, input_shape, out_shape, add_bias_to_output);
        }
    }

    void backprop(Layer* layer) override
    {
        shape temp_out_shape = output_shape;
        if (layer->is_device_layer())
        {
            calcDerivativeWRtoInput(derivativeWRtoInputGPU.get(), input_size, layer->derivative_wr_to_input(), temp_out_shape, weightsGPU.get(), add_bias_to_output);
            if(add_bias_to_output)
                updateWeightsAndBias(weightsGPU.get(), layer->derivative_wr_to_input(), inputPtr, input_size, size - 1, output_shape, learing_rate);
            else
                updateWeightsAndBias(weightsGPU.get(), layer->derivative_wr_to_input(), inputPtr, input_size, size, output_shape, learing_rate);
        }
        else
        {
            cuVector<float> derivativeWRToOutput = layer->get_device_derivative();
            calcDerivativeWRtoInput(derivativeWRtoInputGPU.get(), input_size, derivativeWRToOutput.get(), temp_out_shape, weightsGPU.get(), add_bias_to_output);
            if (add_bias_to_output)
                updateWeightsAndBias(weightsGPU.get(), derivativeWRToOutput.get(), inputPtr, input_size, size - 1, output_shape, learing_rate);
            else
                updateWeightsAndBias(weightsGPU.get(), derivativeWRToOutput.get(), inputPtr, input_size, size, output_shape, learing_rate);
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

    ~LinearLayerGPU()
    {
    }
};