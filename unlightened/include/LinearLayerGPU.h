#pragma once
#include <Layer.h>
#include <vector>
#include <memory>
#include <device_memory.h>

void linearLayerForwardPassGPU(float* output, const float* weights, const float* input, const shape& input_shape, const shape& output_shape, bool bias_subtracted);
void calcDerivativeWRtoInput(float* derivativeWRtoInput, size_t input_size, const float* derivateWRtoOutput, size_t outputSize, const float* weights);
void updateWeightsAndBias(float* weights, const float* derivativeWRtoOutput, const float* input, size_t input_size, size_t outputSize);

class LinearLayerGPU : public Layer
{
private:
    std::vector<float> weight;
    std::vector<float> output;
    std::vector<float> derivativeWRtoInput;
    cuVector<float> weightsGPU;
    cuVector<float> outputGPU;
    cuVector<float> derivativeWRtoInputGPU;
    cuVector<float> inputVectorGPU;
    const float* inputPtr;
    size_t size;
    size_t input_size;
public:

    LinearLayerGPU(size_t neuron_size) : size(neuron_size)
    {
        device_layer = true;
    }

    void init(const shape& input) override
    {
        input_size = input.size();
        weight.resize(input_size * size);
        derivativeWRtoInput.resize(input_size);
        derivativeWRtoInputGPU.setValues(derivativeWRtoInput);
        // +1 is for the bias
        output.resize(size + 1);
        output[size] = 1.0f;
        outputGPU.setValues(output);

        for (size_t i = 0; i < input_size * size; i++)
        {
            weight[i] = 1.0f / (rand() % 1000);
        }
        weightsGPU.setValues(weight);
        size = size + 1;
        output_shape.width = size;
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
        out_shape.width = out_shape.width - 1; // -1 because we dont want to calculate for the bias
        shape input_shape;
        input_shape.width = prevLayer->get_shape().volume(); // represent the value as 1d array
        input_shape.batches = prevLayer->get_shape().batches;
        
        if (!prevLayer->is_device_layer())
        {
            inputVectorGPU = prevLayer->get_device_output();
            inputPtr = inputVectorGPU.get();
            linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr, input_shape, out_shape, true);
        }
        else
        {
            linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr, input_shape, out_shape, true);
        }
    }

    void backprop(Layer* layer) override
    {
        
        if (layer->is_device_layer())
        {
            calcDerivativeWRtoInput(derivativeWRtoInputGPU.get(), input_size, layer->derivative_wr_to_input(), size - 1, weightsGPU.get());
            updateWeightsAndBias(weightsGPU.get(), layer->derivative_wr_to_input(), inputPtr, input_size, size - 1);
        }
        else
        {
            cuVector<float> derivativeWRToOutput = layer->get_device_derivative();
            calcDerivativeWRtoInput(derivativeWRtoInputGPU.get(), input_size, derivativeWRToOutput.get(), size - 1, weightsGPU.get());
            updateWeightsAndBias(weightsGPU.get(), derivativeWRToOutput.get(), inputPtr, input_size, size - 1);
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