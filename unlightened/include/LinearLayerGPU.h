#pragma once
#include <Layer.h>
#include <vector>
#include <memory>
#include <device_memory.h>

void linearLayerForwardPassGPU(float* output, const float* weights, const float* input, const shape& input_shape, const shape& output_shape, bool bias_subtracted);
void calcDerivativeWRtoInput(float* derivativeWRtoInput, size_t input_size, const float* derivateWRtoOutput, size_t outputSize, const float* weights);
void updateWeightsAndBias(float* weights, const float* derivativeWRtoOutput, const float* input, size_t input_size, size_t outputSize);

template <bool val>
class LinearLayerGPU : public Layer
{
private:
    static constexpr bool smh = val;
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

        if (!smh)
        {
            inputVectorGPU.setValues(inputPtr, input_size);
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
        cuVector<float> derivativeWRToOutput;
        derivativeWRToOutput.setValues(layer->derivative_wr_to_input(), size);
        calcDerivativeWRtoInput(derivativeWRtoInputGPU.get(), input_size, derivativeWRToOutput.get(), size - 1, weightsGPU.get());
        updateWeightsAndBias(weightsGPU.get(), derivativeWRToOutput.get(), inputPtr, input_size, size - 1);
    }

    const float* get_output()
    {
        outputGPU.getCopy(output);
        return output.data();
    };
    const float* derivative_wr_to_input()
    {
        derivativeWRtoInputGPU.getCopy(derivativeWRtoInput);
        return derivativeWRtoInput.data();
    }
    void printLayer()
    {
    }

    ~LinearLayerGPU()
    {
    }
};