#pragma once
#include "Layer.h"
#include <vector>
#include <device_memory.h>

void sigmoidLayer(float* input, float* output, size_t input_size);
void sigmoidLayerDerivative(float* derivativeWRtoInput, const float* output, const float* derivativeWRtoOutput, size_t input_size);

class SigmoidLayerGPU : public Layer
{
    std::vector<float> output;
    std::vector<float> derivativeWRtoInput;
    cuVector<float> outputGPU;
    cuVector<float> derivativeWRtoInputGPU;
    cuVector<float> inOutBuffer;
    size_t size;
    size_t input_size;

    void init(const shape& input) override
    {
        input_size = input.size();
        size = input_size;
        output.resize(size);
        output[size - 1] = 1.0f; // this is the bias so its always 1.0f i.e he is alaways fired
        derivativeWRtoInput.resize(input_size);
        outputGPU.setValues(output);
        derivativeWRtoInputGPU.setValues(derivativeWRtoInput);
        output_shape.width = size;
    }

    void forwardPass(Layer* prevLayer) override
    {
        const float* input = prevLayer->getOutput();
        inOutBuffer.setValues(input, input_size);
        sigmoidLayer(inOutBuffer.get(), outputGPU.get(), size - 1); // size -1 because of the bias
        outputGPU.getCopy(output);
    }

    void backprop(Layer* layer) override
    {
        const float* derivativeWRtoOutput = layer->derivativeWithRespectToInput();
        inOutBuffer.setValues(derivativeWRtoOutput, size);
        sigmoidLayerDerivative(derivativeWRtoInputGPU.get(), outputGPU.get(), inOutBuffer.get(), input_size);
        derivativeWRtoInputGPU.getCopy(derivativeWRtoInput);
    }

    const float* getOutput() override
    {
        return output.data();
    }

    const float* derivativeWithRespectToInput() override
    {
        return derivativeWRtoInput.data();
    }

    void printLayer() override
    {
    }
};
