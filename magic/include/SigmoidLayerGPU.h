#pragma once
#include "Layer.h"
#include <vector>
#include <GpuMemory.h>

void sigmoidLayer(float* input, float* output, size_t inputSize);
void sigmoidLayerDerivative(float* derivativeWRtoInput, const float* output, const float* derivativeWRtoOutput, size_t inputSize);

class SigmoidLayerGPU : public Layer
{
    std::vector<float> output;
    std::vector<float> derivativeWRtoInput;
    cuVector<float> outputGPU;
    cuVector<float> derivativeWRtoInputGPU;
    cuVector<float> inOutBuffer;

    void init() override
    {
        size = inputSize;
        output.resize(size);
        output[size - 1] = 1.0f; // this is the bias so its always 1.0f i.e he is alaways fired
        derivativeWRtoInput.resize(inputSize);
        outputGPU.setValues(output);
        derivativeWRtoInputGPU.setValues(derivativeWRtoInput);
    }

    void forwardPass(Layer* prevLayer) override
    {
        const float* input = prevLayer->getOutput();
        inOutBuffer.setValues(input, inputSize);
        sigmoidLayer(inOutBuffer.get(), outputGPU.get(), size - 1); // size -1 because of the bias
        outputGPU.getCopy(output);
    }

    void backprop(Layer* layer) override
    {
        const float* derivativeWRtoOutput = layer->derivativeWithRespectToInput();
        inOutBuffer.setValues(derivativeWRtoOutput, size);
        sigmoidLayerDerivative(derivativeWRtoInputGPU.get(), outputGPU.get(), inOutBuffer.get(), inputSize);
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
