#pragma once
#include "AllLayers.h"
#include <vector>
#include <memory>

class NeuralNet
{
    InputLayer input_layer;
    OutputLayer output_layer;
    std::vector<std::shared_ptr<Layer>> layers;
public:

    NeuralNet(shape input_shape) : input_layer(input_shape)
    {
    }

    void addLayer(Layer* layer)
    {
        layers.push_back(std::shared_ptr<Layer>(layer));
        updateLastLayer();
    }

    void addLayer(std::shared_ptr<Layer>& layer)
    {
        layers.push_back(layer);
        updateLastLayer();
    }

    InputLayer& getInputLayer() noexcept
    {
        return input_layer;
    }

    OutputLayer& loss_layer() noexcept
    {
        return output_layer;
    }

    void predict()
    {
        layers[0]->forward_pass(&input_layer);
        for (size_t i = 1; i < layers.size(); i++)
        {
            layers[i]->forward_pass(layers[i - 1].get());
        }
        output_layer.forward_pass(layers.back().get());
    }

    void backprop(bool calc_loss = true)
    {
        if (calc_loss)
        {
            output_layer.backprop(nullptr);
        }

        layers.back()->backprop(&output_layer);

        for (int i = layers.size() - 2; i >= 0; i--)
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

    std::shared_ptr<Layer>& last()
    {
        return layers.back();
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
            output_layer.init_base(last->get_shape());
        }
        else
        {
            layers.back().get()->init_base(input_layer.get_shape());
        }
    }
};