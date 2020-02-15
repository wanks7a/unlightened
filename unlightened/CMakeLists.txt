add_library(
    unlightened STATIC
    include/AllLayers.h
    include/cnn_layer.h
    include/conv_filter.h
    include/GpuMemory.h
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
    # cpp files
    source/cnn_layer.cpp
    source/cnnLayer.cu
    source/GpuUtils.cpp
    source/LinearLayerGPU.cu
    source/SigmoidLayerGPU.cu
)

target_include_directories(unlightened PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")