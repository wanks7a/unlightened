#pragma once
#include "AllLayers.h"
#include <vector>

class NeuralNet
{
    size_t inputSize;
    std::vector<Layer*> layers;
public:

    NeuralNet(size_t inputLayerSize, bool useBias = true) : inputSize(inputLayerSize)
    {
        addLayer(new InputLayer(inputLayerSize, useBias));
    }

    void addLayer(Layer* layer)
    {
        layers.push_back(layer);
        updateLastLayer();
    }

    InputLayer& getInputLayer() const
    {
        return static_cast<InputLayer&>(*(layers[0]));
    }

    void predict()
    {
        for (size_t i = 1; i < layers.size(); i++)
        {
            layers[i]->forwardPass(layers[i - 1]);
        }
    }
    void backprop()
    {
        layers.back()->backprop(nullptr);

        for (size_t i = layers.size() - 2; i > 0; i--)
        {
            layers[i]->backprop(layers[i + 1]);
        }
    }
    void print()
    {
        for (const auto& l : layers)
        {
            l->printLayer();
            std::cout << std::endl;
        }

    }
    void set_learning_rate(float learning_rate)
    {
        if (learning_rate > 0)
        {
            for (const auto& l : layers)
            {
                l->set_learning_rate(learning_rate);
            }
        }
    }

    ~NeuralNet()
    {
        for (auto& l : layers)
        {
            delete l;
        }
    }
private:
    void updateLastLayer()
    {
        if (layers.size() > 1)
        {
            Layer* last = layers.back();
            Layer* beforeLast = layers[layers.size() - 2];
            last->init(beforeLast->get_shape());
        }
    }
};