#pragma once
#include <vector>
#include <iostream>
#include <serializable_interface.h>

class OutputLayer : public serializable_layer<OutputLayer>
{   
    std::vector<float> predictedValue;
    std::vector<float> derivativeWRToInput;
    std::vector<float> observedValues;
    size_t size;
public:

    OutputLayer() : size(0)
    {
    }

    void init(const shape& input) override
    {
        output_shape = input;
        size = input.size();
        derivativeWRToInput.resize(size);
        observedValues.resize(size);
    }

    void forward_pass(Layer* prevLayer) override
    {
        predictedValue = prevLayer->get_native_output();
    };

    void backprop(Layer* layer) override
    {
        for (size_t i = 0; i < size; i++)
        {
            derivativeWRToInput[i] = -2 * (observedValues[i] - predictedValue[i]);
        }
    }

    bool setObservedValue(const std::vector<float>& observedVal)
    {
        if (observedVal.size() == observedValues.size())
        {
            observedValues = observedVal;
            return true;
        }
        return false;
    }
    void set_derivative_manual(const std::vector<float>& deriv)
    {
        if (derivativeWRToInput.size() == deriv.size())
            derivativeWRToInput = deriv;
        else
            std::cout << "Manual derivative size is wrong " << std::endl;
    }
    void print_predicted(size_t values = 0)
    {
        if (values > size || values == 0)
            values = size;
        //std::cout << "Values :" << std::endl;
        for (int i = 0; i < values; i++)
        {
            printf("Value [%d] = %.2f      Actual = %.2f \n", i, predictedValue[i], observedValues[i]);
        }
    }
    const float* get_output() const override
    {
        return predictedValue.data();
    }
    const float* derivative_wr_to_input() const override
    {
        return derivativeWRToInput.data();
    }

    float get_total_loss() const
    {
        float result = 0;
        for (size_t i = 0; i < size; i++)
        {
            result += ((observedValues[i] - predictedValue[i]) * (observedValues[i] - predictedValue[i]));
        }
        return result;
    }

    template <typename Serializer>
    void serialize_members(Serializer& s) const
    {
    }

    template <typename Serializer>
    void deserialize_members(Serializer& s)
    {
        init(input_shape);
    }

    ~OutputLayer() = default;
};
