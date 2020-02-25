#include <conv_filter.h>
#include <GpuUtils.h>
#include <device_memory.h>
#include <LinearLayer.h>
#include <NeuralNet.h>
#include <SigmoidLayerGPU.h>
#include <cnn_layer.h>
#include <LinearLayerGPU.h>
#include <array>
#include <unordered_map>
#include <gtest/gtest.h>
#include <max_pool.h>
#include <activation_layer.h>
#include <cmath>
#include <csv_parser.h>

class gpu_tests : public ::testing::Test 
{
protected:
    //virtual void SetUp() 
    //{
    //    ASSERT_TRUE(utils::GpuInit());
    //}

    //virtual void TearDown() 
    //{
    //    ASSERT_TRUE(utils::GpuRelase());
    //}
};

struct test_layer : public Layer
{
    test_layer()
    {
        device_layer = true;
    }

    cuVector<float> output;
    void init(const shape& input) override {};
    void forward_pass(Layer* prevLayer) override {};
    void backprop(Layer* layer) override {};
    const float* get_output()  override
    {
        return output.get();
    };
    const float* derivative_wr_to_input() override
    {
        return output.get();
    };

    void printLayer() override
    {
    }

    void set_output_shape(const shape& sh)
    {
        output_shape = sh;
    }
};
//#include <chrono>
//TEST(gpu_tests, mnist_d)
//{
//    NeuralNet net(28 * 28);
//    csv<float> mnist("C:\\Users\\wanks7a\\Desktop\\mnist_train.csv");
//
//    std::unordered_map<float, std::array<float, 10>> mm =
//    {
//        {0.0f, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0} },
//        {1.0f, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0} },
//        {2.0f, {0, 0, 1, 0, 0, 0, 0, 0, 0, 0} },
//        {3.0f, {0, 0, 0, 1, 0, 0, 0, 0, 0, 0} },
//        {4.0f, {0, 0, 0, 0, 1, 0, 0, 0, 0, 0} },
//        {5.0f, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0} },
//        {6.0f, {0, 0, 0, 0, 0, 0, 1, 0, 0, 0} },
//        {7.0f, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0} },
//        {8.0f, {0, 0, 0, 0, 0, 0, 0, 0, 1, 0} },
//        {9.0f, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1} },
//    };
//    net.getInputLayer().set_output_shape(shape(28, 28));
//    net.addLayer(new LinearLayerGPU<false>(128));
//    net.addLayer(new SigmoidLayerGPU());
//    net.addLayer(new LinearLayerGPU<false>(128));
//    net.addLayer(new SigmoidLayerGPU());
//    net.addLayer(new LinearLayerGPU<false>(10));
//    net.addLayer(new SigmoidLayerGPU());
//    OutputLayer* loss = new OutputLayer();
//    net.addLayer(loss);
//
//    for (size_t i = 0; i < mnist.rows.size(); i++)
//    {
//        net.getInputLayer().setInput(mnist.rows[i].elements.data() + 1, 28 * 28);
//        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//
//        net.predict();
//        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
//        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
//        //auto it = mm.find(mnist.rows[i].elements[0]);
//        //net.
//        //net.backprop()
//    }
//}
//

TEST(gpu_tests, backprop_filter_test)
{
    shape input_shape(4, 4, 2, 2);
    shape output_shape(4, 4, 3, 2); // 3 filters
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1, 1, 1, 1,
        0, 0, 0, 0,
        1, 1, 1, 1,
        0, 0, 0, 0,
        1, 1, 1, 1,
        0, 0, 0, 0,
        1, 1, 1, 1,
        0, 0, 0, 0,
        1, 1, 1, 1,
        0, 0, 0, 0,
        1, 1, 1, 1,
        0, 0, 0, 0,
        1, 1, 1, 1,
        0, 0, 0, 0,
        1, 1, 1, 1,
        0, 0, 0, 0,
        }));
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 1,
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 1,
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 1,
        }));

    std::vector<float> expected = {
        2, 2, 1,
        1, 2, 2,
        1, 1, 1,
        2, 2, 1,
        1, 2, 2,
        1, 1, 1,
        1, 1, 0,
        0, 1, 1,
        0, 0, 0,
        1, 1, 0,
        0, 1, 1,
        0, 0, 0
        };

    shape weights_deriv(3, 3, 2, 2);
    cuVector<float> output;
    output.resize(weights_deriv.size());
    for (size_t i = 0; i < 3; i++) // 3 because there are 3 filters
    {
        full_conv_2d(input.get(), input_shape, output.get(), weights_deriv, weights.get() + i * output_shape.area(), output_shape.width, output_shape.height, 3, output_shape.volume());
        std::vector<float> result;
        output.getCopy(result);
        EXPECT_TRUE(result.size(), expected.size());
        for (size_t j = 0; j < result.size(); j++)
        {
            EXPECT_EQ(result[i], expected[i]);
        }
    }
}

TEST(gpu_tests, cnn_2filters5x5_1_1batch)
{
    test_layer  test;
    test.init_base(shape(5, 5, 1));
    test.set_output_shape(shape(5, 5, 1));
    test.output.setValues({
        2,2,2,1,2,
        0,2,1,2,2,
        1,1,0,1,0,
        2,1,1,1,2,
        1,1,2,2,2
        });
    cnn_layer cnn_l(3, 3);
    auto opt = cnn_l.get_options();
    opt.zeropadding = false;
    cnn_l.set_options(opt);
    cnn_l.init_base(test.get_shape());

    EXPECT_TRUE(cnn_l.get_filters().get_weights().setValues({
        0,-1,-1,
        0,0,-1,
        0, 1, 1,
        0,-1,-1,
        0,0,-1,
        0, 1, 1,
        0,-1,-1,
        0,0,-1,
        0, 1, 1,
        }));
    

    cnn_l.forward_pass(&test);

    std::vector<float> expected = {
        -3, -3, -3,
        0, -1, 0,
        2, 3, 2,
        -3, -3, -3,
        0, -1, 0,
        2, 3, 2,
        -3, -3, -3,
        0, -1, 0,
        2, 3, 2,
    };

    std::vector<float> result = cnn_l.get_native_output();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, cnn_2filters5x5_2depth_1batch)
{
    test_layer  test;
    test.init_base(shape(5, 5, 2));
    test.set_output_shape(shape(5, 5, 2));
    test.output.setValues({
        2,2,2,1,2,
        0,2,1,2,2,
        1,1,0,1,0,
        2,1,1,1,2,
        1,1,2,2,2,
        2,2,2,1,2,
        0,2,1,2,2,
        1,1,0,1,0,
        2,1,1,1,2,
        1,1,2,2,2,
        });
    cnn_layer cnn_l(3,3);
    auto opt = cnn_l.get_options();
    opt.zeropadding = false;
    cnn_l.set_options(opt);
    cnn_l.init_base(test.get_shape());

    EXPECT_TRUE(cnn_l.get_filters().get_weights().setValues({
        0,-1,-1,
        0,0,-1,
        0, 1, 1,
        0,-1,-1,
        0,0,-1,
        0, 1, 1,
        0,-1,-1,
        0,0,-1,
        0, 1, 1,
        0,-1,-1,
        0,0,-1,
        0, 1, 1,
        0,-1,-1,
        0,0,-1,
        0, 1, 1,
        0,-1,-1,
        0,0,-1,
        0, 1, 1,
        }));
    

    cnn_l.forward_pass(&test);
 
    std::vector<float> expected = { 
        -6, -6, -6,
        0, -2, 0,
        4, 6, 4,
        -6, -6, -6,
        0, -2, 0,
        4, 6, 4,
        -6, -6, -6,
        0, -2, 0,
        4, 6, 4,
    };

    std::vector<float> result = cnn_l.get_native_output();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, cnn_2filters5x5_2depth_2batch)
{
    test_layer  test;
    test.init_base(shape(5, 5, 2, 2));
    test.set_output_shape(shape(5, 5, 2, 2));
    test.output.setValues({
        2,2,2,1,2,
        0,2,1,2,2,
        1,1,0,1,0,
        2,1,1,1,2,
        1,1,2,2,2,
        2,2,2,1,2,
        0,2,1,2,2,
        1,1,0,1,0,
        2,1,1,1,2,
        1,1,2,2,2,
        2,2,2,1,2,
        0,2,1,2,2,
        1,1,0,1,0,
        2,1,1,1,2,
        1,1,2,2,2,
        2,2,2,1,2,
        0,2,1,2,2,
        1,1,0,1,0,
        2,1,1,1,2,
        1,1,2,2,2,
        });
    cnn_layer cnn_l(3, 3);
    auto opt = cnn_l.get_options();
    opt.zeropadding = false;
    cnn_l.set_options(opt);
    cnn_l.init_base(test.get_shape());

    EXPECT_TRUE(cnn_l.get_filters().get_weights().setValues({
        0,-1,-1,
        0,0,-1,
        0, 1, 1,
        0,-1,-1,
        0,0,-1,
        0, 1, 1,
        0,-1,-1,
        0,0,-1,
        0, 1, 1,
        0,-1,-1,
        0,0,-1,
        0, 1, 1,
        0,-1,-1,
        0,0,-1,
        0, 1, 1,
        0,-1,-1,
        0,0,-1,
        0, 1, 1,
        }));
    

    cnn_l.forward_pass(&test);

    std::vector<float> expected = {
        -6, -6, -6,
        0, -2, 0,
        4, 6, 4,
        -6, -6, -6,
        0, -2, 0,
        4, 6, 4,
        -6, -6, -6,
        0, -2, 0,
        4, 6, 4,
        -6, -6, -6,
        0, -2, 0,
        4, 6, 4,
        -6, -6, -6,
        0, -2, 0,
        4, 6, 4,
        -6, -6, -6,
        0, -2, 0,
        4, 6, 4,
    };

    std::vector<float> result = cnn_l.get_native_output();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, merge_conv_with_bias_2d)
{
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,1,1,
        2,2,2,
        3,3,3,
        1,1,1,
        2,2,2,
        3,3,3,
        }));
    shape input_shape(3, 3, 2);
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({
        1,2
        }));
    cuVector<float> output;
    output.resize(input_shape.area());
    merge_conv_with_bias(input.get(), input_shape, bias.get(), output.get(), 0);
    std::vector<float> expected = {
        5,5,5,
        7,7,7,
        9,9,9
    };
    std::vector<float> result;
    output.getCopy(result);

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, merge_conv_with_bias_2d_batch2)
{
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,1,1,
        2,2,2,
        3,3,3,
        1,1,1,
        2,2,2,
        3,3,3,
        1,1,1,
        2,2,2,
        3,3,3,
        1,1,1,
        2,2,2,
        3,3,3,
        }));
    shape input_shape(3, 3, 2, 2);
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({
        1,2
        }));
    cuVector<float> output;
    output.resize(input_shape.area() * input_shape.batches);
    merge_conv_with_bias(input.get(), input_shape, bias.get(), output.get(), input_shape.area());
    std::vector<float> expected = {
        5,5,5,
        7,7,7,
        9,9,9,
        5,5,5,
        7,7,7,
        9,9,9,
    };
    std::vector<float> result;
    output.getCopy(result);

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, merge_conv_with_bias_2d_batch2_bigger_offset)
{
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,1,1,
        2,2,2,
        3,3,3,
        1,1,1,
        2,2,2,
        3,3,3,
        1,1,1,
        2,2,2,
        3,3,3,
        1,1,1,
        2,2,2,
        3,3,3,
        }));
    shape input_shape(3, 3, 2, 2);
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({
        1,2
        }));
    cuVector<float> output;
    output.resize(input_shape.area() * input_shape.batches + 3);
    merge_conv_with_bias(input.get(), input_shape, bias.get(), output.get(), input_shape.area() + 3);
    std::vector<float> expected = {
        5,5,5,
        7,7,7,
        9,9,9,
        0,0,0,
        5,5,5,
        7,7,7,
        9,9,9,
    };
    std::vector<float> result;
    output.getCopy(result);

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, activation_2d)
{
    test_layer  test;
    test.init_base(shape(3, 3));
    test.set_output_shape(shape(3, 3));
    EXPECT_TRUE(test.output.setValues({
        1,1,1,
        2,2,2,
        3,3,3
        }));
    activation_layer act(activation_layer::activation_function::Identity);
    act.init_base(test.get_shape());
    act.forward_pass(&test);
    std::vector<float> expected = {
        1,1,1,
        2,2,2,
        3,3,3
    };

    std::vector<float> result = act.get_native_output();
   
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
    EXPECT_TRUE(test.output.setValues({
        5,5,5,
        6,6,6,
        7,7,7
        }));
    expected = {
        5,5,5,
        6,6,6,
        7,7,7
    };
    act.backprop(&test);
    result = act.get_native_derivative();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}


TEST(gpu_tests, activation_3d)
{
    test_layer  test;
    test.init_base(shape(3, 3, 2));
    test.set_output_shape(shape(3, 3, 2));
    test.output.setValues({
        1,1,1,
        2,2,2,
        3,3,3,
        5,5,5,
        6,6,6,
        7,7,7
        });
    activation_layer act(activation_layer::activation_function::Identity);
    act.init_base(test.get_shape());
    act.forward_pass(&test);
    std::vector<float> expected = {
        1,1,1,
        2,2,2,
        3,3,3,
        5,5,5,
        6,6,6,
        7,7,7
    };

    std::vector<float> result = act.get_native_output();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }

    EXPECT_TRUE(test.output.setValues({
        5,5,5,
        6,6,6,
        7,7,7,
        8,8,8,
        9,9,9,
        1,1,1
        }));
    expected = {
        5,5,5,
        6,6,6,
        7,7,7,
        8,8,8,
        9,9,9,
        1,1,1
    };
    act.backprop(&test);
    result = act.get_native_derivative();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, activation_3d_batch2)
{
    test_layer  test;
    test.init_base(shape(3, 3, 2, 2));
    test.set_output_shape(shape(3, 3, 2, 2));
    test.output.setValues({
        1,1,1,
        2,2,2,
        3,3,3,
        5,5,5,
        6,6,6,
        7,7,7,
        10,10,10,
        11,11,11,
        12,12,12,
        13,13,13,
        14,14,14,
        15,15,15
        });
    activation_layer act(activation_layer::activation_function::Identity);
    act.init_base(test.get_shape());
    act.forward_pass(&test);
    std::vector<float> expected = {
        1,1,1,
        2,2,2,
        3,3,3,
        5,5,5,
        6,6,6,
        7,7,7,
        10,10,10,
        11,11,11,
        12,12,12,
        13,13,13,
        14,14,14,
        15,15,15
    };

    std::vector<float> result = act.get_native_output();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }

    EXPECT_TRUE(test.output.setValues({
        5,5,5,
        6,6,6,
        7,7,7,
        8,8,8,
        9,9,9,
        1,1,1,
        5,5,5,
        6,6,6,
        7,7,7,
        8,8,8,
        9,9,9,
        1,1,1
        }));
    expected = {
        5,5,5,
        6,6,6,
        7,7,7,
        8,8,8,
        9,9,9,
        1,1,1,
        5,5,5,
        6,6,6,
        7,7,7,
        8,8,8,
        9,9,9,
        1,1,1
    };
    act.backprop(&test);
    result = act.get_native_derivative();

    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, max_pool_back_prop_depth1)
{
    shape input_shape(3, 3);
    shape output_shape(5, 5);

    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,2,3,
        4,5,6,
        7,8,9
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size(), 20.0f));
    cuVector<char> mask;
    mask.setValues({
        0, 1, 2,
        3, 0, 2,
        1, 0, 0
        });

    std::vector<float> expected = {
        1, 0, 0, 2, 0,
        0, 0, 0, 0, 3,
        0, 0, 5, 0, 0,
        0, 4, 0, 0, 6,
        0, 7, 8, 0, 9
    };

    max_pool_backprop(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
    std::vector<float> result;
    output.getCopy(result);
    EXPECT_EQ(expected.size(), result.size());
    for (size_t i = 0; i < expected.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
    }
}

TEST(gpu_tests, max_pool_back_prop_depth2)
{
    shape input_shape(3, 3, 2);
    shape output_shape(5, 5, 2);

    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,2,3,
        4,5,6,
        7,8,9,
        9,8,7,
        6,5,4,
        3,2,1,
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size(), 20.0f));
    cuVector<char> mask;
    mask.setValues({
        0, 1, 2,
        3, 0, 2,
        1, 0, 0,
        1, 3, 2,
        2, 0, 0,
        1, 1, 0
        });

    std::vector<float> expected = {
        1, 0, 0, 2, 0,
        0, 0, 0, 0, 3,
        0, 0, 5, 0, 0,
        0, 4, 0, 0, 6,
        0, 7, 8, 0, 9,
        0, 9, 0, 0, 0,
        0, 0, 0, 8, 7,
        0, 0, 5, 0, 4,
        6, 0, 0, 0, 0,
        0, 3, 0, 2, 1
    };

    max_pool_backprop(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
    std::vector<float> result;
    output.getCopy(result);
    EXPECT_EQ(expected.size(), result.size());
    for (size_t i = 0; i < expected.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
    }
}

TEST(gpu_tests, max_pool_back_prop_depth2_batch2)
{
    shape input_shape(3, 3, 2, 2);
    shape output_shape(5, 5, 2, 2);

    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,2,3,
        4,5,6,
        7,8,9,
        9,8,7,
        6,5,4,
        3,2,1,
        9,8,7,
        6,5,4,
        3,2,1,
        1,2,3,
        4,5,6,
        7,8,9
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size(), 20.0f)); 
    cuVector<char> mask;
    mask.setValues({
        0, 1, 2,
        3, 0, 2,
        1, 0, 0,
        1, 3, 2,
        2, 0, 0,
        1, 1, 0,
        1, 3, 2,
        2, 0, 0,
        1, 1, 0,
        0, 1, 2,
        3, 0, 2,
        1, 0, 0
        });

    std::vector<float> expected = {
        1, 0, 0, 2, 0,
        0, 0, 0, 0, 3,
        0, 0, 5, 0, 0,
        0, 4, 0, 0, 6,
        0, 7, 8, 0, 9,
        0, 9, 0, 0, 0,
        0, 0, 0, 8, 7,
        0, 0, 5, 0, 4,
        6, 0, 0, 0, 0,
        0, 3, 0, 2, 1,
        0, 9, 0, 0, 0,
        0, 0, 0, 8, 7,
        0, 0, 5, 0, 4,
        6, 0, 0, 0, 0,
        0, 3, 0, 2, 1,
        1, 0, 0, 2, 0,
        0, 0, 0, 0, 3,
        0, 0, 5, 0, 0,
        0, 4, 0, 0, 6,
        0, 7, 8, 0, 9
    };

    max_pool_backprop(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
    std::vector<float> result;
    output.getCopy(result);
    EXPECT_EQ(expected.size(), result.size());
    for (size_t i = 0; i < expected.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
    }
}

TEST(gpu_tests, max_pool_depth1)
{
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,0,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,1,0,0,0,
        0,0,0,0,0
        }));
    shape input_shape;
    input_shape.width = 5;
    input_shape.height = 5;
    input_shape.depth = 1;
    shape output_shape;
    output_shape.width = 3;
    output_shape.height = 3;
    output_shape.depth = 1;
    cuVector<char> mask;
    EXPECT_TRUE(mask.resize(output_shape.size()));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size(), 0.0f));

    std::vector<float> expected =
    {
        1, 1, 0,
        1, 1, 0,
        0, 0, 0
    };
    std::vector<int> mask_expected =
    {
        0, 2, 0,
        3, 1, 0,
        0, 0, 0
    };
    max_pool(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<char> mask_result;
    mask.getCopy(mask_result);
    EXPECT_EQ(expected.size(), result.size());
    EXPECT_EQ(mask_expected.size(), mask_result.size());
    for (size_t i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
        EXPECT_EQ(mask_expected[i], mask_result[i]);
    }
}

TEST(gpu_tests, max_pool_depth2)
{
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,0,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,1,0,0,0,
        0,0,0,0,0,
        0,1,0,0,0,
        0,0,0,1,0,
        0,0,0,0,0,
        0,1,1,0,0,
        0,0,0,0,0
    }));
    shape input_shape;
    input_shape.width = 5;
    input_shape.height = 5;
    input_shape.depth = 2;
    shape output_shape;
    output_shape.width = 3;
    output_shape.height = 3;
    output_shape.depth = 2;
    cuVector<char> mask;
    EXPECT_TRUE(mask.resize(output_shape.size()));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size(), 0.0f));

    std::vector<float> expected =
    {
        1, 1, 0,
        1, 1, 0,
        0, 0, 0,
        1, 1, 0,
        1, 1, 0,
        0, 0, 0
    };
    std::vector<int> mask_expected =
    {
        0, 2, 0,
        3, 1, 0,
        0, 0, 0,
        1, 3, 0,
        3, 2, 0,
        0, 0, 0
    };
    max_pool(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<char> mask_result;
    mask.getCopy(mask_result);
    EXPECT_EQ(expected.size(), result.size());
    EXPECT_EQ(mask_expected.size(), mask_result.size());
    for (size_t i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
        EXPECT_EQ(mask_expected[i], mask_result[i]);
    }
}

TEST(gpu_tests, max_pool_depth2_batch2)
{
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,0,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,1,0,0,0,
        0,0,0,0,0,
        0,1,0,0,0,
        0,0,0,1,0,
        0,0,0,0,0,
        0,1,1,0,0,
        0,0,0,0,0,
        1,0,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,1,0,0,0,
        0,0,0,0,0,
        0,1,0,0,0,
        0,0,0,1,0,
        0,0,0,0,0,
        0,1,1,0,0,
        0,0,0,0,0,
        }));
    shape input_shape;
    input_shape.width = 5;
    input_shape.height = 5;
    input_shape.depth = 2;
    input_shape.batches = 2;
    shape output_shape;
    output_shape.width = 3;
    output_shape.height = 3;
    output_shape.depth = 2;
    output_shape.batches = 2;
    cuVector<char> mask;
    EXPECT_TRUE(mask.resize(output_shape.size()));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size(), 0.0f));

    std::vector<float> expected =
    {
        1, 1, 0,
        1, 1, 0,
        0, 0, 0,
        1, 1, 0,
        1, 1, 0,
        0, 0, 0,
        1, 1, 0,
        1, 1, 0,
        0, 0, 0,
        1, 1, 0,
        1, 1, 0,
        0, 0, 0,
    };
    std::vector<int> mask_expected =
    {
        0, 2, 0,
        3, 1, 0,
        0, 0, 0,
        1, 3, 0,
        3, 2, 0,
        0, 0, 0,
        0, 2, 0,
        3, 1, 0,
        0, 0, 0,
        1, 3, 0,
        3, 2, 0,
        0, 0, 0,
    };
    max_pool(input.get(), input_shape, output.get(), output_shape, mask.get(), 2);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<char> mask_result;
    mask.getCopy(mask_result);
    EXPECT_EQ(expected.size(), result.size());
    EXPECT_EQ(mask_expected.size(), mask_result.size());
    for (size_t i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
        EXPECT_EQ(mask_expected[i], mask_result[i]);
    }
}

TEST(gpu_tests, cnn_conv_3d_depth1_same)
{
    shape input_shape;
    input_shape.width = 5;
    input_shape.height = 5;
    shape output_shape;
    output_shape.width = 5;
    output_shape.height = 5;
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2
    }));
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1
        }));
    cuVector<float> output;
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({ 0 }));
    EXPECT_TRUE(output.resize(output_shape.area()));
    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), bias.get(), 3, 3, 2);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<float> expected = { 0, 1, 2, 2, 2,
                                  -4, -4, -4, -4, -2,
                                    0, -1, -2, -1, 0,
                                    -1, 1, 2, 1, 2,
                                     -4, -4, -4, -5, -2 };
    EXPECT_EQ(expected.size(), result.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(expected[i], result[i]);
    }
}

TEST(gpu_tests, cnn_conv_3d_depth3_same)
{
    shape input_shape;
    input_shape.width = 5;
    input_shape.height = 5;
    input_shape.depth = 3;
    shape output_shape;
    output_shape.width = 5;
    output_shape.height = 5;
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2
        }));
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1,
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1,
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1
        }));
    cuVector<float> output;
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({ 0, 0, 0 }));
    EXPECT_TRUE(output.resize(output_shape.size()));
    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), bias.get(), 3, 3, 2);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<float> expected = { 
                        0, 1, 2, 2, 2,
                        -4, -4, -4, -4, -2,
                        0, -1, -2, -1, 0,
                        -1, 1, 2, 1, 2,
                        -4, -4, -4, -5, -2
    };
    EXPECT_EQ(expected.size(), result.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ((expected[i] * input_shape.depth), result[i]);
    }
}

TEST(gpu_tests, cnn_conv_3d_depth3_batch2_same)
{
    shape input_shape;
    input_shape.width = 5;
    input_shape.height = 5;
    input_shape.depth = 3;
    input_shape.batches = 2;
    shape output_shape;
    output_shape.width = 5;
    output_shape.height = 5;
    output_shape.depth = 1;
    output_shape.batches = 2;
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
        }));
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1,
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1,
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size()));
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({ 0, 0, 0 }));
    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), bias.get(), 3, 3, 2);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<float> expected = {
                        0, 1, 2, 2, 2,
                        -4, -4, -4, -4, -2,
                        0, -1, -2, -1, 0,
                        -1, 1, 2, 1, 2,
                        -4, -4, -4, -5, -2,
                        0, 1, 2, 2, 2,
                        -4, -4, -4, -4, -2,
                        0, -1, -2, -1, 0,
                        -1, 1, 2, 1, 2,
                        -4, -4, -4, -5, -2
    };
    EXPECT_EQ(expected.size(), result.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(expected[i] * input_shape.depth, result[i]);
    }
}

TEST(gpu_tests, cnn_conv_3d_depth3_batch2_same_v2)
{
    shape input_shape;
    input_shape.width = 5;
    input_shape.height = 5;
    input_shape.depth = 3;
    input_shape.batches = 2;
    shape output_shape;
    output_shape.width = 5;
    output_shape.height = 5;
    output_shape.depth = 2;
    output_shape.batches = 2;
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
                                 2,2,2,1,2,
                                 0,2,1,2,2,
                                 1,1,0,1,0,
                                 2,1,1,1,2,
                                 1,1,2,2,2,
        }));
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1,
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1,
                        0,-1,-1,
                        0,0,-1,
                        0, 1, 1
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size()));
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({ 0, 0, 0 }));
    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), bias.get(), 3, 3, 2);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<float> expected = {
                        0, 1, 2, 2, 2,
                        -4, -4, -4, -4, -2,
                        0, -1, -2, -1, 0,
                        -1, 1, 2, 1, 2,
                        -4, -4, -4, -5, -2,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0, 1, 2, 2, 2,
                        -4, -4, -4, -4, -2,
                        0, -1, -2, -1, 0,
                        -1, 1, 2, 1, 2,
                        -4, -4, -4, -5, -2,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
    };
    EXPECT_EQ(expected.size(), result.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(expected[i] * input_shape.depth, result[i]);
    }
}

TEST(gpu_tests, cnn_conv_3d_depth1_valid)
{
    shape input_shape;
    input_shape.width = 6;
    input_shape.height = 6;
    shape output_shape;
    output_shape.width = 4;
    output_shape.height = 4;
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
    }));
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({
                        1,1,1,
                        0,0,0,
                        -1,-1,-1
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(static_cast<size_t>(output_shape.width) * output_shape.height));
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({ 0, 0, 0 }));
    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), bias.get(), 3, 3, 3);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<float> expected = { 0,0,0,0,
                                    30,10,-10,-30,
                                    30,10,-10,-30,
                                    0,0,0,0
    };
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, cnn_conv_3d_depth3_valid)
{
    shape input_shape;
    input_shape.width = 6;
    input_shape.height = 6;
    input_shape.depth = 3;
    shape output_shape;
    output_shape.width = 4;
    output_shape.height = 4;
    output_shape.depth = 1;
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
        }));
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({
                        1,1,1,
                        0,0,0,
                        -1,-1,-1,
                        1,1,1,
                        0,0,0,
                        -1,-1,-1,
                        1,1,1,
                        0,0,0,
                        -1,-1,-1
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.volume()));
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({ 0, 0, 0 }));
    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), bias.get(), 3, 3, 3);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<float> expected = { 0,0,0,0,
                                    30,10,-10,-30,
                                    30,10,-10,-30,
                                    0,0,0,0
    };
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(expected[i] * input_shape.depth, result[i]);
    }
}

TEST(gpu_tests, cnn_conv_3d_depth3_batch2_valid)
{
    shape input_shape;
    input_shape.width = 6;
    input_shape.height = 6;
    input_shape.depth = 3;
    input_shape.batches = 2;
    shape output_shape;
    output_shape.width = 4;
    output_shape.height = 4;
    output_shape.depth = 1;
    output_shape.batches = 2;
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        10,10,10,0,0,0,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
                        0,0,0,10,10,10,
        }));
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({
                        1,1,1,
                        0,0,0,
                        -1,-1,-1,
                        1,1,1,
                        0,0,0,
                        -1,-1,-1,
                        1,1,1,
                        0,0,0,
                        -1,-1,-1
        }));
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size()));
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({ 0, 0, 0 }));
    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), bias.get(), 3, 3, 3);
    std::vector<float> result;
    output.getCopy(result);
    std::vector<float> expected = { 0,0,0,0,
                                    30,10,-10,-30,
                                    30,10,-10,-30,
                                    0,0,0,0,
                                    0,0,0,0,
                                    30,10,-10,-30,
                                    30,10,-10,-30,
                                    0,0,0,0,
    };
    EXPECT_EQ(expected.size(), result.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(expected[i] * input_shape.depth, result[i]);
    }
}

TEST(gpu_tests, cnn_full_conv_2d_001)
{
    shape input_shape(5, 5);
    shape output_shape(6, 6);
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({ 1, 1, 1, 1 }));
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5
        }));

    cuVector<float> output;
    EXPECT_TRUE(output.resize(6 * 6));
    std::vector<float> expected = {
        1,3,5,7,9,5,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        1,3,5,7,9,5
    };
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({ 0 }));
    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), bias.get(), 2, 2, 1);
    std::vector<float> result;
    output.getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(gpu_tests, cnn_full_conv_2d_002)
{
    shape input_shape(5, 5, 3);
    shape output_shape(6, 6, 1);
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({ 
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1
        }));
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5
        }));

    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size()));
    std::vector<float> expected = {
        1,3,5,7,9,5,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        1,3,5,7,9,5
    };
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({ 0, 0, 0 }));
    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), bias.get(), 2, 2, 1);
    std::vector<float> result;
    output.getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i] * input_shape.depth);
    }
}

TEST(gpu_tests, cnn_full_conv_2d_003)
{
    shape input_shape(5, 5, 3, 2);
    shape output_shape(6, 6, 1, 2);
    cuVector<float> weights;
    EXPECT_TRUE(weights.setValues({
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1
        }));
    cuVector<float> input;
    EXPECT_TRUE(input.setValues({
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        6,7,8,9,10,
        1,2,3,4,5,
        }));

    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size()));
    std::vector<float> expected = {
        1,3,5,7,9,5,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        1,3,5,7,9,5,
        1,3,5,7,9,5,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        7,16,20,24,28,15,
        1,3,5,7,9,5
    };
    cuVector<float> bias;
    EXPECT_TRUE(bias.setValues({ 0, 0, 0 }));

    conv_3d(input.get(), input_shape, output.get(), output_shape, weights.get(), bias.get(), 2, 2, 1);
    std::vector<float> result;
    output.getCopy(result);
    EXPECT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(result[i], expected[i] * input_shape.depth);
    }
}

TEST(gpu_tests, cnn_basic)
{
    shape input_shape;
    input_shape.width = 7;
    input_shape.height = 7;
    cnn_layer layer(3, 5);
    auto options = layer.get_options();
    options.zeropadding = false;
    layer.set_options(options);
    layer.init(input_shape);
    EXPECT_EQ(layer.get_shape().depth, 5);
    EXPECT_EQ(layer.get_shape().height, 5);
    EXPECT_EQ(layer.get_shape().width, 5);
}

TEST(gpu_tests, dense_layer1024)
{
    std::vector<float> input;
    input.resize(1024 * 1024, 1.0f);
    LinearLayerGPU<false> l(1024);
    shape input_shape(1, 1024);
    l.init(input_shape);
    EXPECT_EQ(l.get_shape().size(), 1024 + 1); // because of bias
    EXPECT_TRUE(l.set_weights(input));
    cuVector<float> inputPtr;
    inputPtr.setValues(input);
    cuVector<float> weightsGPU;
    weightsGPU.setValues(input);
    cuVector<float> outputGPU;
    EXPECT_TRUE(outputGPU.resize(1025));
    std::vector<float> result;
    for (size_t i = 0; i < 1000; i++)
    {
        linearLayerForwardPassGPU(outputGPU.get(), weightsGPU.get(), inputPtr.get(), 1024, 1024, true);
    }
    outputGPU.getCopy(result);
    for (size_t i = 0; i < result.size()-1; i++)
    {
        EXPECT_EQ(1024, result[i]);
    }
}

TEST(gpu_tests, dense_layer1024_batches2_v1)
{
    shape input_shape(1024, 1, 1, 2);
    cuVector<float> input;
    EXPECT_TRUE(input.resize(input_shape.size(), 1.0f));
    shape output_shape(1024, 1, 1, 2);
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size()));
    cuVector<float> weights;
    EXPECT_TRUE(weights.resize(1024 * 1024, 1.0f));
    std::vector<float> result;
    linearLayerForwardPassGPU(output.get(), weights.get(), input.get(), input_shape, output_shape, false);
    output.getCopy(result);
    EXPECT_EQ(result.size(), output_shape.size());
    for (size_t i = 0; i < result.size(); i++)
    {
        EXPECT_EQ(1024, result[i]);
    }
}

TEST(gpu_tests, dense_layer1024_batches2_v2)
{
    shape input_shape(1024, 1, 1, 2);
    cuVector<float> input;
    EXPECT_TRUE(input.resize(input_shape.size(), 1.0f));
    shape output_shape(1025, 1, 1, 2);
    cuVector<float> output;
    EXPECT_TRUE(output.resize(output_shape.size()));
    cuVector<float> weights;
    EXPECT_TRUE(weights.resize(1024 * 1024, 1.0f));
    std::vector<float> result;
    linearLayerForwardPassGPU(output.get(), weights.get(), input.get(), input_shape, shape(1024,1,1,2), true);
    output.getCopy(result);
    EXPECT_EQ(result.size(), output_shape.size());
    for (size_t i = 0; i < 1024; i++)
    {
        EXPECT_EQ(1024, result[i]);
    }
    EXPECT_EQ(0, result[1024]);
    for (size_t i = 1025; i < result.size() - 1; i++)
    {
        EXPECT_EQ(1024, result[i]);
    }
    EXPECT_EQ(0, result[result.size() - 1]);
}

TEST(gpu_tests, test_xor_cpu)
{
    NeuralNet test(2, true);
    test.addLayer(new LinearLayer(2));
    test.addLayer(new SigmoidLayer());
    test.addLayer(new LinearLayer(1));
    test.addLayer(new SigmoidLayer());
    OutputLayer* loss = new OutputLayer();
    test.addLayer(loss);
    test.set_learning_rate(0.01f);
    for (int i = 0; i < 1000000; i++)
    {
        test.getInputLayer().setInput(std::array<float, 2>{0, 1}.data(), 2);
        test.predict();
        loss->setObservedValue({ 1,0 });
        test.backprop();
        test.getInputLayer().setInput(std::array<float, 2>{1, 1}.data(), 2);
        test.predict();
        loss->setObservedValue({ 0,0 });
        test.backprop();
        test.getInputLayer().setInput(std::array<float, 2>{0, 0}.data(), 2);
        test.predict();
        loss->setObservedValue({ 0,0 });
        test.backprop();
        test.getInputLayer().setInput(std::array<float, 2>{1, 0}.data(), 2);
        test.predict();
        loss->setObservedValue({ 1,0 });
        test.backprop();
    }
    test.getInputLayer().setInput(std::array<float, 2>{1, 0}.data(), 2);
    test.predict();
    EXPECT_GT(loss->get_output()[0], 0.95f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
    test.getInputLayer().setInput(std::array<float, 2>{0, 0}.data(), 2);
    test.predict();
    EXPECT_LT(loss->get_output()[0], 0.05f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
    test.getInputLayer().setInput(std::array<float, 2>{1, 1}.data(), 2);
    test.predict();
    EXPECT_LT(loss->get_output()[0], 0.05f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
    test.getInputLayer().setInput(std::array<float, 2>{0, 1}.data(), 2);
    test.predict();
    EXPECT_GT(loss->get_output()[0], 0.95f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
}

TEST(gpu_tests, test_xor_gpu)
{
    NeuralNet test(2, true);
    test.addLayer(new LinearLayerGPU<false>(10));
    test.addLayer(new SigmoidLayerGPU());
    test.addLayer(new LinearLayerGPU<false>(1));
    test.addLayer(new SigmoidLayerGPU());
    OutputLayer* loss = new OutputLayer();
    test.addLayer(loss);
    test.set_learning_rate(0.01f);
    for (int i = 0; i < 10000; i++)
    {
        test.getInputLayer().setInput(std::array<float, 2>{0, 1}.data(), 2);
        test.predict();
        loss->setObservedValue({ 1,0 });
        test.backprop();
        test.getInputLayer().setInput(std::array<float, 2>{1, 1}.data(), 2);
        test.predict();
        loss->setObservedValue({ 0,0 });
        test.backprop();
        test.getInputLayer().setInput(std::array<float, 2>{0, 0}.data(), 2);
        test.predict();
        loss->setObservedValue({ 0,0 });
        test.backprop();
        test.getInputLayer().setInput(std::array<float, 2>{1, 0}.data(), 2);
        test.predict();
        loss->setObservedValue({ 1,0 });
        test.backprop();
    }
    test.getInputLayer().setInput(std::array<float, 2>{1, 0}.data(), 2);
    test.predict();
    EXPECT_GT(loss->get_output()[0], 0.9f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
    test.getInputLayer().setInput(std::array<float, 2>{0, 0}.data(), 2);
    test.predict();
    EXPECT_LT(loss->get_output()[0], 0.1f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
    test.getInputLayer().setInput(std::array<float, 2>{1, 1}.data(), 2);
    test.predict();
    EXPECT_LT(loss->get_output()[0], 0.1f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
    test.getInputLayer().setInput(std::array<float, 2>{0, 1}.data(), 2);
    test.predict();
    EXPECT_GT(loss->get_output()[0], 0.9f);
    EXPECT_EQ(loss->get_output()[1], 1.0f);
}

int main(int argc, char** argv)
{
    EXPECT_TRUE(utils::GpuInit());
    ::testing::InitGoogleTest(&argc, argv);
    EXPECT_TRUE(utils::GpuRelase());
    return RUN_ALL_TESTS();
}