#pragma once
#include "AllLayers.h"
#include <vector>
#include <memory>

class NeuralNet
{
    std::vector<std::shared_ptr<Layer>> layers;
public:

    NeuralNet(shape input_shape)
    {
        addLayer(new InputLayer(input_shape));
    }

    void addLayer(Layer* layer, bool update = true)
    {
        layers.push_back(std::shared_ptr<Layer>(layer));
        if(update)
            updateLastLayer();
    }

    void addLayer(std::shared_ptr<Layer>& layer, bool update = true)
    {
        layers.push_back(layer);
        if(update)
            updateLastLayer();
    }

    InputLayer& getInputLayer() const
    {
        return static_cast<InputLayer&>(*((layers[0].get())));
    }

    void predict()
    {
        for (size_t i = 1; i < layers.size(); i++)
        {
            layers[i]->forward_pass(layers[i - 1].get());
        }
    }
    void backprop(bool calc_loss = true)
    {
        if(calc_loss)
            layers.back()->backprop(nullptr);

        for (size_t i = layers.size() - 2; i > 0; i--)
        {
            layers[i]->backprop(layers[i + 1].get());
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

    void set_update_weights(bool flag)
    {
        for (size_t i = 0; i < layers.size(); i++)
        {
            layers[i]->set_update_weights(flag);
        }
    }

    std::shared_ptr<Layer> operator[](size_t index)
    {
        if (index < layers.size())
        {
            return layers[index];
        }

        return nullptr;
    }

    ~NeuralNet()
    {
    }
private:
    void updateLastLayer()
    {
        if (layers.size() > 1)
        {
            Layer* last = layers.back().get();
            Layer* beforeLast = layers[layers.size() - 2].get();
            last->init_base(beforeLast->get_shape());
        }
    }
};