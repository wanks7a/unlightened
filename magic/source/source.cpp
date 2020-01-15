#include <NeuralNet.h>
#include <array>
#include <LinearLayerGPU.h>
#include <SigmoidLayerGPU.h>
#include <GpuUtils.h>

int main()
{
    if (!utils::GpuInit())
        return 0;
    NeuralNet test(2, true);
    test.addLayer(new LinearLayerGPU<false>(10));
    //test.addLayer(new LinearLayer(10));
    //test.addLayer(new SigmoidLayer());
    test.addLayer(new SigmoidLayerGPU());
    test.addLayer(new LinearLayer(1));
    //test.addLayer(new SigmoidLayer());
    test.addLayer(new SigmoidLayerGPU());
    OutputLayer loss;
    test.addLayer(&loss);
    for (int i = 0; i < 10000; i++)
    {
        test.getInputLayer().setInput(std::array<float, 2>{0, 1}.data(), 2);
        test.predict();
        loss.setObservedValue({ 1,0 });
        test.backprop();
        test.getInputLayer().setInput(std::array<float, 2>{1, 1}.data(), 2);
        test.predict();
        loss.setObservedValue({ 0,0 });
        test.backprop();
        test.getInputLayer().setInput(std::array<float, 2>{0, 0}.data(), 2);
        test.predict();
        loss.setObservedValue({ 0,0 });
        test.backprop();
        test.getInputLayer().setInput(std::array<float, 2>{1, 0}.data(), 2);
        test.predict();
        loss.setObservedValue({ 1,0 });
        test.backprop();
    }
    test.getInputLayer().setInput(std::array<float, 2>{1, 0}.data(), 2);
    test.predict();
    loss.printLayer();
    test.getInputLayer().setInput(std::array<float, 2>{0, 0}.data(), 2);
    test.predict();
    loss.printLayer();
    test.getInputLayer().setInput(std::array<float, 2>{1, 1}.data(), 2);
    test.predict();
    loss.printLayer();
    test.getInputLayer().setInput(std::array<float, 2>{0, 1}.data(), 2);
    test.predict();
    loss.printLayer();
    utils::GpuRelase();
    return 0;
}