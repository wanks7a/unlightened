add_library(
    unlightened STATIC
    include/AllLayers.h
    include/cnn_layer.h
    include/conv_filter.h
    include/device_memory.h
    include/GpuUtils.h
    include/InputLayer.h
    include/Layer.h
    include/LinearLayer.h
    include/LinearLayerGPU.h
    include/NeuralNet.h
    include/OutputLayer.h
    include/shape.h
    include/SigmoidLayer.h
    include/SigmoidLayerGPU.h
    include/max_pool.h
    include/activation_layer.h
    # cpp files
    source/cnn_layer.cpp
    source/cnnLayer.cu
    source/GpuUtils.cpp
    source/LinearLayerGPU.cu
    source/SigmoidLayerGPU.cu
    source/max_pool.cu
    source/activation_layer.cpp
    source/activation_layer.cu
)

target_include_directories(unlightened PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")