#pragma once
#include <Layer.h>
#include <vector>
#include <memory>
#include <GpuMemory.h>

void linearLayerForwardPassGPU(float* output, const float* weights,const float* input, size_t inputSize, size_t outputSize);
void calcDerivativeWRtoInput(float* derivativeWRtoInput, size_t inputSize, const float* derivateWRtoOutput, size_t outputSize, const float* weights);
void updateWeightsAndBias(float* weights, const float* derivativeWRtoOutput, const float* input, size_t inputSize, size_t outputSize);

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
public:

    LinearLayerGPU(size_t neuronSize, bool includeBias = true)
    {
        size = neuronSize;
    }

    void init() override
    {
        weight.resize(inputSize * size);
        derivativeWRtoInput.resize(inputSize);
        derivativeWRtoInputGPU.setValues(derivativeWRtoInput);
        // +1 is for the bias
        output.resize(size + 1);
        output[size] = 1.0f;
        outputGPU.setValues(output);

        for (size_t i = 0; i < inputSize * size; i++)
        {
            weight[i] = 1.0f / (rand() % 1000);
        }
        weightsGPU.setValues(weight);
        size = size + 1;
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

    void forwardPass(Layer* prevLayer) override
    {
        inputPtr = prevLayer->getOutput();
        if (!smh)
        {
            inputVectorGPU.setValues(inputPtr, inputSize);
            inputPtr = inputVectorGPU.get();
            linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr, inputSize, size - 1);
        }
        else
        {
            linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr, inputSize, size - 1);
        }
    }

    void backprop(Layer* layer) override
    {
        cuVector<float> derivativeWRToOutput;
        derivativeWRToOutput.setValues(layer->derivativeWithRespectToInput(), size);
        calcDerivativeWRtoInput(derivativeWRtoInputGPU.get(), inputSize, derivativeWRToOutput.get(), size - 1, weightsGPU.get());
        updateWeightsAndBias(weightsGPU.get(), derivativeWRToOutput.get(), inputPtr, inputSize, size - 1);
    }

    const float* getOutput()
    {
        outputGPU.getCopy(output);
        return output.data();
    };
    const float* derivativeWithRespectToInput()
    {
        derivativeWRtoInputGPU.getCopy(derivativeWRtoInput);
        return derivativeWRtoInput.data();
    }
    void printLayer()
    {
    }
};