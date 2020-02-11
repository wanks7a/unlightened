#pragma once
#include <Layer.h>
#include <vector>
#include <memory>
#include <GpuMemory.h>

void linearLayerForwardPassGPU(float* output, const float* weights,const float* input, size_t input_size, size_t outputSize);
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

    void forwardPass(Layer* prevLayer) override
    {
        inputPtr = prevLayer->getOutput();
        if (!smh)
        {
            inputVectorGPU.setValues(inputPtr, input_size);
            inputPtr = inputVectorGPU.get();
            linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr, input_size, size - 1);
        }
        else
        {
            linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr, input_size, size - 1);
        }
    }

    void backprop(Layer* layer) override
    {
        cuVector<float> derivativeWRToOutput;
        derivativeWRToOutput.setValues(layer->derivativeWithRespectToInput(), size);
        calcDerivativeWRtoInput(derivativeWRtoInputGPU.get(), input_size, derivativeWRToOutput.get(), size - 1, weightsGPU.get());
        updateWeightsAndBias(weightsGPU.get(), derivativeWRToOutput.get(), inputPtr, input_size, size - 1);
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