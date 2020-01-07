#pragma once

class Layer
{
protected:
    size_t size;
    size_t inputSize;
    static constexpr float learing_rate = 0.01f;
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
    size_t getOutputSize() const { return size; }
    size_t getInputSize() const { return inputSize; }
};