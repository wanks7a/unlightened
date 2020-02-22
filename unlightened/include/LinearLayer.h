#pragma once
#include <vector>
#include <iostream>
#include "Layer.h"

class LinearLayer : public Layer
{
private:
    std::vector<float> weight;
    std::vector<float> output;
    std::vector<float> derivativeWRtoInput;
    const float* input;
    size_t size;
    size_t input_size;
public:
    LinearLayer(size_t neuron_size) : input(nullptr), input_size(0), size(neuron_size)
    {
    }

    void init(const shape& input) override
    {
        input_size = input.size();
        weight.resize(input_size * size);
        derivativeWRtoInput.resize(input_size);
        // +1 is for the bias
        output.resize(size + 1);
        output[size] = 1.0f;
        for (size_t i = 0; i < input_size * size; i++)
        {
            weight[i] = 1.0f / (rand() % 1000);
        }
        size = size + 1;
        output_shape.width = size;
    }

    void forward_pass(Layer* prevLayer) override
    {
        input = prevLayer->get_output();
        forwardPassCalcNeurons(0, size - 1, prevLayer);
        
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
            for (size_t j = 0; j < input_size; j++)
            {
                output[i] += input[j] * weight[i * input_size + j];
            }
        }
    }

    void backprop(Layer* layer) override
    {
        const float* derivativeWRToOutput = layer->derivative_wr_to_input();
        calcDerivativeWRtoInput(0, input_size, derivativeWRToOutput);
        updateWeightsAndBias(0, size - 1, derivativeWRToOutput);
    }

    void updateWeightsAndBias(size_t neuronIndex, const float* derivativeWRtoOutput)
    {
        for (size_t i = 0; i < input_size; i++)
        {
            weight[neuronIndex * input_size + i] = weight[neuronIndex * input_size + i] - learing_rate * input[i] * derivativeWRtoOutput[neuronIndex];
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
            derivativeWRtoInput[inputIndex] += derivateWRtoOutput[i] * weight[i * input_size + inputIndex];
        }
    }

    void calcDerivativeWRtoInput(size_t inputIndexFrom, size_t count, const float* derivateWRtoOutput)
    {
        for (size_t i = inputIndexFrom; i < inputIndexFrom + count; i++)
        {
            calcDerivativeWRtoInput(i, derivateWRtoOutput);
        }
    }

    const float* get_output() override
    {
        return output.data();
    }

    const float* derivative_wr_to_input() override
    {
        return this->derivativeWRtoInput.data();
    }

    void printLayer() override
    {
        std::cout << "Input : " << input_size << " Output: " << size << std::endl;
        for (size_t i = 0; i < size - 1; i++)
        {
            std::cout << "Neuron : " << i << std::endl;
            for (size_t j = 0; j < input_size; j++)
            {
                std::cout << "weight[" << i << "]" << "[" << j << "] = " << weight[i * input_size + j] << " " << std::endl;
            }
        }

    }

    ~LinearLayer()
    {
    }
};