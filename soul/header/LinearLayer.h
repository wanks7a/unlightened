#pragma once
#include <vector>
#include <thread>
#include <iostream>
#include "Layer.h"

class LinearLayer : public Layer
{
private:
    std::vector<float> weight;
    std::vector<float> output;
    std::vector<float> derivativeWRtoInput;
    const float* input;
    float useBias;
public:
    LinearLayer(size_t neuronSize, bool includeBias = true)
    {
        useBias = includeBias;
        size = neuronSize;
    }

    void init() override
    {
        weight.resize(inputSize * size);
        derivativeWRtoInput.resize(inputSize);
        // +1 is for the bias
        output.resize(size + 1);
        output[size] = 1.0f;
        for (size_t i = 0; i < inputSize * size; i++)
        {
            weight[i] = 1.0f / (rand() % 1000);
        }
        size = size + 1;
    }

    void forwardPass(Layer* prevLayer) override
    {
        unsigned int n = std::thread::hardware_concurrency();
        unsigned int neuronsPerThread = (size - 1) / n;
        input = prevLayer->getOutput();
        if (neuronsPerThread == 0)
        {
            forwardPassCalcNeurons(0, size - 1, prevLayer);
        }
        else
        {
            std::vector<std::thread> threads;
            for (size_t i = 0; i < n - 1; i++)
            {
                threads.push_back(
                    std::thread(&LinearLayer::forwardPassCalcNeurons, this, i * neuronsPerThread, neuronsPerThread, prevLayer));
            }
            unsigned int remainingNeurons = (size - 1) % n;
            forwardPassCalcNeurons((n - 1) * neuronsPerThread, neuronsPerThread + remainingNeurons, prevLayer);
            for (auto& tr : threads)
            {
                if (tr.joinable())
                    tr.join();
            }
        }
        if (output[size - 1] != 1.0f)
        {
            output[size - 1] = 1.0f;
        }
    }

    void forwardPassCalcNeurons(size_t offset, size_t numberOfNeurons, Layer* prevLayer)
    {
        if (offset > size || (offset + numberOfNeurons) > size)
            return;

        for (size_t i = offset; i < offset + numberOfNeurons; i++)
        {
            this->output[i] = 0.0f;
            for (size_t j = 0; j < inputSize; j++)
            {
                output[i] += input[j] * weight[i * inputSize + j];
            }
        }
    }

    void backprop(Layer* layer) override
    {
        const float* derivativeWRToOutput = layer->derivativeWithRespectToInput();
        calcDerivativeWRtoInput(0, inputSize, derivativeWRToOutput);
        updateWeightsAndBias(0, size - 1, derivativeWRToOutput);
    }

    void updateWeightsAndBias(size_t neuronIndex, const float* derivativeWRtoOutput)
    {
        for (size_t i = 0; i < inputSize; i++)
        {
            weight[neuronIndex * inputSize + i] = weight[neuronIndex * inputSize + i] - learing_rate * input[i] * derivativeWRtoOutput[neuronIndex];
        }
    }

    void updateWeightsAndBias(size_t fromNeuronIndex, size_t count, const float* derivativeWRtoOutput)
    {
        for (size_t i = fromNeuronIndex; i < fromNeuronIndex + count; i++)
        {
            updateWeightsAndBias(i, derivativeWRtoOutput);
        }
    }

    void calcDerivativeWRtoInput(size_t inputIndex, const float* derivateWRtoOutput)
    {
        derivativeWRtoInput[inputIndex] = 0.0f;
        // size - 1 because the last neuron is actually the bias 
        // which does not depenend from the input layer
        for (size_t i = 0; i < size - 1; i++)
        {
            derivativeWRtoInput[inputIndex] += derivateWRtoOutput[i] * weight[i * inputSize + inputIndex];
        }
    }

    void calcDerivativeWRtoInput(size_t inputIndexFrom, size_t count, const float* derivateWRtoOutput)
    {
        for (size_t i = inputIndexFrom; i < inputIndexFrom + count; i++)
        {
            calcDerivativeWRtoInput(i, derivateWRtoOutput);
        }
    }

    const float* getOutput() override
    {
        return output.data();
    }

    const float* derivativeWithRespectToInput() override
    {
        return this->derivativeWRtoInput.data();
    }

    void printLayer() override
    {
        std::cout << "Input : " << inputSize << " Output: " << size << std::endl;
        for (size_t i = 0; i < size - 1; i++)
        {
            std::cout << "Neuron : " << i << std::endl;
            for (size_t j = 0; j < inputSize; j++)
            {
                std::cout << "weight[" << i << "]" << "[" << j << "] = " << weight[i * inputSize + j] << " " << std::endl;
            }
        }

    }
};