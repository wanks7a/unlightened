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
    include/shape.h
    include/SigmoidLayer.h
    include/SigmoidLayerGPU.h
    include/max_pool.h
    include/activation_layer.h
    include/csv_parser.h
    include/loss_layer.h
    include/generic_functions.h
    include/softmax.h
    include/reshape_layer.h
    include/binary_serialization.h
    include/gan.h
    include/conv_transpose.h
    include/conv2d_cudnn.h
    include/generic_stream.h
    include/serializable_interface.h
    include/serializer_interface.h
    include/cudnn_helpers.h
    include/batch_runner.h
    include/optimizer.h
    include/device_vector.h
    include/cuda_device.h
    include/optimizer_momentum.h
    include/device_layer.h
    include/batch_norm.h
    include/batch_norm_cuda.h
    include/read_write_stream.h
    include/cuda_vector_norm.h
    include/adam_optimizer.h
    include/cuda_vector.h
    include/blob.h
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
    source/reshape_layer.cpp
    source/gan.cpp
    source/conv_transpose.cpp
    source/conv2d_cudnn.cpp
    source/binary_serialization.cpp
    source/batch_runner.cpp
    source/optimizer.cpp
    source/optimizer_momentum.cpp
    source/batch_norm.cpp
    source/batch_norm_cuda.cpp
    source/read_write_stream.cpp
    source/cuda_vector_norm.cu
    source/adam_optimizer.cpp
    source/cuda_vector.cpp
)

find_path(CUDNN_INCLUDE cudnn.h)

find_library(CUDNN_LIB cudnn)
message("CUDNN Headers= " ${CUDNN_INCLUDE} )
message("CUDNN LIB location= " ${CUDNN_LIB})

target_compile_options(unlightened PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)
target_include_directories(unlightened PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" ${CUDNN_INCLUDE})
target_link_libraries(unlightened ${CUDNN_LIB})