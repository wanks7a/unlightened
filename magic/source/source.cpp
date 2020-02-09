#include <NeuralNet.h>
#include <array>
#include <LinearLayerGPU.h>
#include <SigmoidLayerGPU.h>
#include <GpuUtils.h>
#include <conv_filter.h>

int main()
{
    if (!utils::GpuInit())
        return 0;
    //NeuralNet test(2, true);
    ////test.addLayer(new LinearLayer(10));
    ////test.addLayer(new SigmoidLayer());
    //test.addLayer(new LinearLayerGPU<false>(10));
    //test.addLayer(new SigmoidLayerGPU());
    //test.addLayer(new LinearLayer(1));
    ////test.addLayer(new SigmoidLayer());
    //test.addLayer(new SigmoidLayerGPU());
    //OutputLayer loss;
    //test.addLayer(&loss);
    //for (int i = 0; i < 10000; i++)
    //{
    //    test.getInputLayer().setInput(std::array<float, 2>{0, 1}.data(), 2);
    //    test.predict();
    //    loss.setObservedValue({ 1,0 });
    //    test.backprop();
    //    test.getInputLayer().setInput(std::array<float, 2>{1, 1}.data(), 2);
    //    test.predict();
    //    loss.setObservedValue({ 0,0 });
    //    test.backprop();
    //    test.getInputLayer().setInput(std::array<float, 2>{0, 0}.data(), 2);
    //    test.predict();
    //    loss.setObservedValue({ 0,0 });
    //    test.backprop();
    //    test.getInputLayer().setInput(std::array<float, 2>{1, 0}.data(), 2);
    //    test.predict();
    //    loss.setObservedValue({ 1,0 });
    //    test.backprop();
    //}
    //test.getInputLayer().setInput(std::array<float, 2>{1, 0}.data(), 2);
    //test.predict();
    //loss.printLayer();
    //test.getInputLayer().setInput(std::array<float, 2>{0, 0}.data(), 2);
    //test.predict();
    //loss.printLayer();
    //test.getInputLayer().setInput(std::array<float, 2>{1, 1}.data(), 2);
    //test.predict();
    //loss.printLayer();
    //test.getInputLayer().setInput(std::array<float, 2>{0, 1}.data(), 2);
    //test.predict();
    //loss.printLayer();

    filter_options opt(2,2);
    filter_conv2d filter(opt);
    shape input_shape;
    input_shape.width = 5;
    input_shape.height = 8;
    std::vector<float> input = { 1,2,3,4,5,
                                 1,2,3,4,5,
                                 1,2,3,4,5,
                                 1,2,3,4,5,
                                 1,2,3,4,5,
                                 1,2,3,4,5,
                                 1,2,3,4,5,
                                 1,2,3,4,5};
    std::vector<float> output;
    cuVector<float> inputK;
    inputK.setValues(input);
    cuVector<float> outputK;
    output.resize(10*7);
    outputK.setValues(output);
    //filter_forwardPass(inputK.get(), input_shape, 0, outputK.get(), );
    outputK.getCopy(output);
    utils::GpuRelase();
    return 0;
}