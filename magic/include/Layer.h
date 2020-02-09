#pragma once

class Layer
{
protected:
    size_t size;
    size_t inputSize;
    float learing_rate = 0.1f;
public:
    virtual void init() = 0;
    virtual void forwardPass(Layer* prevLayer) = 0;
    virtual void backprop(Layer* layer) = 0;
    virtual const float* getOutput() = 0;
    virtual const float* derivativeWithRespectToInput() = 0;
    virtual void printLayer() = 0;
    void setInputSize(size_t inpSize)
    {
        inputSize = inpSize;
    }
    void set_learning_rate(float rate)
    {
        if (rate < 1.0f && rate > 0)
        {
            learing_rate = rate;
        }
    }
    size_t getOutputSize() const { return size; }
    size_t getInputSize() const { return inputSize; }
};