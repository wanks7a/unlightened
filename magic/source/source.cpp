#include <NeuralNet.h>
#include <array>

int main()
{
    NeuralNet test(2, true);
    test.addLayer(new LinearLayer(2));
    test.addLayer(new SigmoidLayer());
    test.addLayer(new LinearLayer(1));
    test.addLayer(new SigmoidLayer());
    OutputLayer loss;
    test.addLayer(&loss);
    for (int i = 0; i < 1000000; i++)
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
    return 0;
}