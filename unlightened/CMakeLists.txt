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
    include/csv_parser.h
    include/loss_layer.h
    include/generic_functions.h
    include/softmax.h
    # cpp files
    source/cnn_layer.cpp
    source/cnnLayer.cu
    source/GpuUtils.cpp
    source/LinearLayerGPU.cu
    source/SigmoidLayerGPU.cu
    source/max_pool.cu
    source/activation_layer.cpp
    source/activation_layer.cu
    source/max_pool.cpp
    source/loss_layer.cu
    source/loss_layer.cpp
    source/generic_functions.cu
    source/softmax.cu
    source/softmax.cpp
)

find_path(CUDNN_INCLUDE cudnn.h)

find_library(CUDNN_LIB cudnn)
message("CUDNN Headers= " ${CUDNN_INCLUDE} )
message("CUDNN LIB location= " ${CUDNN_LIB})

target_compile_options(unlightened PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)
target_include_directories(unlightened PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" ${CUDNN_INCLUDE})
target_link_libraries(unlightened ${CUDNN_LIB})