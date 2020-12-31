#pragma once
#include "AllLayers.h"
#include <vector>
#include <memory>
#include <optimizer.h>

class model
{
    InputLayer input_layer;
    OutputLayer output_layer;
    std::vector<std::shared_ptr<Layer>> layers;
    std::vector<std::shared_ptr<optimizer>> optimizers;

public:

    model(shape input_shape) : input_layer(input_shape)
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
        optimizers.back()->update(layers.back().get());

        for (int i = layers.size() - 2; i >= 0; i--)
        {
            layers[i]->backprop(layers[i + 1].get());
            optimizers[i]->update(layers[i].get());
        }
    }

    void pre_epoch(size_t e)
    {
        for (size_t i = 1; i < layers.size(); i++)
        {
            layers[i]->pre_epoch(e);
        }
    }

    void post_epoch(size_t e)
    {
        for (size_t i = 1; i < layers.size(); i++)
        {
            layers[i]->post_epoch(e);
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

    template <typename Serializer>
    void serialize(Serializer& s) const
    {
        s << layers.size();
        for (size_t i = 0; i < layers.size(); i++)
        {
            s.serialize(*layers[i]);
        }
        s.serialize(input_layer);
        s.serialize(output_layer);
    }

    template <typename Serializer>
    bool load(Serializer& s)
    {
        size_t size;
        s >> size;
        if (!layers.empty())
            layers.clear();
        for (size_t i = 0; i < size; i++)
        {
            layers.push_back(s.deserialize_layer());
        }
        s.deserialize(input_layer);
        s.deserialize(output_layer);
        return true;
    }

    template <typename Serializer>
    bool reload(Serializer& s)
    {
        size_t size;
        s >> size;
        if (layers.size() != size)
            return false;
        for (size_t i = 0; i < size; i++)
        {
            if (!s.deserialize(*layers[i]))
                return false;
        }
        s.deserialize(input_layer);
        s.deserialize(output_layer);
        return true;
    }

    template <typename T, typename ...Args>
    void set_optimizer(Args&& ... args)
    {
        optimizers.clear();
        for (size_t i = 0; i < layers.size(); i++)
        {
            std::shared_ptr<optimizer> o(new T(std::forward<Args>(args)...));
            o->init(layers[i].get());
            optimizers.emplace_back(o);
        }
    }

    ~model()
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