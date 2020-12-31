#pragma once
#include <Layer.h>
#include <vector>
#include <memory>
#include <device_memory.h>
#include <math.h>
#include <serializable_interface.h>

void linearLayerForwardPassGPU(float* output, const float* weights, const float* input, const shape& input_shape, const float* bias, const shape& output_shape);
void calcDerivativeWRtoInput(float* derivativeWRtoInput, size_t inputSize, const float* derivateWRtoOutput, shape output_shape, const float* weights);
void calcWeightsDeriv(const float* weights, float* weights_deriv, const float* derivativeWRtoOutput, const float* input, size_t input_size, size_t outputSize, shape out_shape);
void calcBiasDeriv(const float* bias, float* bias_deriv, const float* derivative_wr_to_out, size_t output_size, shape output_shape);

class dense_gpu : public serializable_layer<dense_gpu>
{
private:
    std::vector<float> weights;
    cuVector<float> weightsGPU;
    cuVector<float> biasGPU;
    cuVector<float> outputGPU;
    cuVector<float> derivativeWRtoInputGPU;
    cuVector<float> inputVectorGPU;
    cuVector<float> weights_deriv;
    cuVector<float> bias_deriv;
    const float* inputPtr;
    size_t size;
    size_t input_size;
public:

    dense_gpu() : size(0), inputPtr(nullptr), input_size(0)
    {
    }

    dense_gpu(size_t neuron_size) : size(neuron_size), inputPtr(nullptr), input_size(0)
    {
        in_device_memory = true;
    }

    void init(const shape& input) override
    {
        input_size = input.volume();
        weights.resize(input_size * size);
        derivativeWRtoInputGPU.resize(input.size(), 0.0f);
        biasGPU.resize(size, 0.0f);
        outputGPU.resize(size * input.batches, 0.0f);
        weightsGPU.setValues(weights);
        weightsGPU.randomize();
        float fan_in = static_cast<float>(input_size);
        weightsGPU *= sqrtf(2.0f / fan_in);
        weights_deriv.resize(weightsGPU.size(), 0.0f);
        bias_deriv.resize(biasGPU.size(), 0.0f);
     
        output_shape.width = size;
        output_shape.batches = input_shape.batches;
    }

    bool set_weights(const std::vector<float>& w)
    {
        if (weights.size() == w.size())
        {
            weights = w;
            return weightsGPU.setValues(weights);
        }
        return false;
    }

    void forward_pass(Layer* prevLayer) override
    {
        inputPtr = prevLayer->get_output();
        shape out_shape = output_shape;

        shape input_shape;
        input_shape.width = prevLayer->get_shape().volume(); // represent the value as 1d array
        input_shape.batches = prevLayer->get_shape().batches;
        if (!prevLayer->is_device_layer())
        {
            inputVectorGPU = prevLayer->get_device_output();
            inputPtr = inputVectorGPU.get();
            linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr, input_shape, biasGPU.get(), out_shape);
        }
        else
        {
            linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr, input_shape, biasGPU.get(), out_shape);
        }
    }

    void backprop(Layer* layer) override
    {
        shape temp_out_shape = output_shape;
        if (layer->is_device_layer())
        {
            calcDerivativeWRtoInput(derivativeWRtoInputGPU.get(), input_size, layer->derivative_wr_to_input(), temp_out_shape, weightsGPU.get());
            if (update_on_backprop)
            {
                calcWeightsDeriv(weightsGPU.get(), weights_deriv.get(), layer->derivative_wr_to_input(), inputPtr, input_size, size, output_shape);
                calcBiasDeriv(biasGPU.get(), bias_deriv.get(), layer->derivative_wr_to_input(), size, output_shape);
            }
        }
        else
        {
            cuVector<float> derivativeWRToOutput = layer->get_device_derivative();
            calcDerivativeWRtoInput(derivativeWRtoInputGPU.get(), input_size, derivativeWRToOutput.get(), temp_out_shape, weightsGPU.get());
            if (update_on_backprop)
            {
                calcWeightsDeriv(weightsGPU.get(), weights_deriv.get(), derivativeWRToOutput.get(), inputPtr, input_size, size, output_shape);
                calcBiasDeriv(biasGPU.get(), bias_deriv.get(), derivativeWRToOutput.get(), size, output_shape);
            }
        }
    }

    const float* get_output() const override
    {
        return outputGPU.get();
    };
    const float* derivative_wr_to_input() const override
    {
        return derivativeWRtoInputGPU.get();
    }

    template <typename Serializer>
    void serialize_members(Serializer& s) const
    {
        s << size << input_size << weightsGPU << biasGPU;
    }

    template <typename Serializer>
    void deserialize_members(Serializer& s)
    {
        s >> size >> input_size >> weightsGPU >> biasGPU;
        outputGPU.resize(size * input_shape.batches, 0.0f);
        derivativeWRtoInputGPU.resize(input_shape.size(), 0.0f);
        weights_deriv.resize(weightsGPU.size(), 0.0f);
        bias_deriv.resize(biasGPU.size(), 0.0f);
    }

    weights_properties get_weights() const override
    {
        weights_properties props;
        props.size = weightsGPU.size();
        props.ptr = weightsGPU.get();
        return props;
    };

    weights_properties get_weights_deriv() const override
    {
        weights_properties props;
        props.size = weights_deriv.size();
        props.ptr = weights_deriv.get();
        return props;
    };

    weights_properties get_bias() const override
    {
        weights_properties props;
        props.size = biasGPU.size();
        props.ptr = biasGPU.get();
        return props;
    };

    weights_properties get_bias_deriv() const override
    {
        weights_properties props;
        props.size = bias_deriv.size();
        props.ptr = bias_deriv.get();
        return props;
    };

    ~dense_gpu()
    {
    }
};