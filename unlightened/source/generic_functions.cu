#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <generic_functions.h>
#include <GpuUtils.h>
#include <vector>

template <unsigned int NUM_OF_THREADS> __global__
void k_sum_all(const float* input, float* value, size_t max_size)
{
    __shared__ float sh_memory[NUM_OF_THREADS];
    const unsigned int tr_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tr_index >= max_size)
    {
        sh_memory[threadIdx.x] = 0.0f;
        return;
    }
    else
    {
        sh_memory[threadIdx.x] = input[tr_index];
    }

    __syncthreads();
    #pragma unroll
    for (int i = NUM_OF_THREADS / 2; i > 0; i = i / 2)
    {
        if (threadIdx.x < i)
            sh_memory[threadIdx.x] += sh_memory[threadIdx.x + i];
        __syncthreads();
    }
    
    if(threadIdx.x == 0)
        atomicAdd(value, sh_memory[threadIdx.x]);
}

void sum_all_values(const shape& sh, const float* input, float* value)
{
    int num_of_threads = 1024;
    auto blocks = utils::getBlockSize(num_of_threads, sh.volume());
    std::vector<cudaStream_t> streams;
    streams.resize(sh.batches);
    for (size_t i = 0; i < sh.batches; i++)
    {
        cudaStreamCreate(&streams[i]);
        k_sum_all<1024> << <blocks, num_of_threads, 0, streams[i] >> > (input + i * sh.volume(), value + i, sh.volume());
    }
    utils::waitAndCheckForErrors();
    for (size_t i = 0; i < sh.batches; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
}

__global__
void k_vec_divide(float* vec1, float* vec2, size_t max_size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < max_size; i += blockDim.x * gridDim.x)
    {
        vec1[i] = vec1[i] / vec2[i];
    }
}

void vector_divide(float* vec1, float* vec2, size_t size)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    k_vec_divide <<<32 * numSMs, 256 >> > (vec1, vec2, size);
    utils::waitAndCheckForErrors();
}

struct MulOp
{
    __host__ __device__
        inline float operator()(float a, float b) const
    {
        return a * b;
    }
};

struct AddOp
{
    __host__ __device__
        inline float operator()(float a, float b) const
    {
        return a + b;
    }
};

struct SqrtOp
{
    __device__
        inline float operator()(float a, float b) const
    {
        return __fsqrt_rn(a);
    }
};


template <typename Func> __global__
void k_vec_scalar_op(float* vec1, size_t max_size, float val)
{
    constexpr Func func;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < max_size; i += blockDim.x * gridDim.x)
    {
        vec1[i] = func(vec1[i], val);
    }
}

void vector_scale(float* vec1, size_t size, float value)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    k_vec_scalar_op<MulOp> << <32 * numSMs, 256 >> > (vec1, size, value);
    utils::waitAndCheckForErrors();
}

void vector_add(float* vec1, size_t size, float value)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    k_vec_scalar_op<AddOp> << <32 * numSMs, 256 >> > (vec1, size, value);
    utils::waitAndCheckForErrors();
}

template <typename Func> __global__
void k_vec_elemwise_op(float* vec1, const float* vec2, size_t max_size)
{
    constexpr Func func;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < max_size; i += blockDim.x * gridDim.x)
    {
        vec1[i] = func(vec1[i], vec2[i]);
    }
}

void vector_mul(float* vec1, const float* vec2, size_t size)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    k_vec_elemwise_op<MulOp> << <32 * numSMs, 256 >> > (vec1, vec2, size);
    utils::waitAndCheckForErrors();
}

void vector_add(float* vec1, const float* vec2, size_t size)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    k_vec_elemwise_op<AddOp> << <32 * numSMs, 256 >> > (vec1, vec2, size);
    utils::waitAndCheckForErrors();
}

void vector_sqrt(float* vec1, size_t size)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    k_vec_scalar_op<SqrtOp> << <32 * numSMs, 256 >> > (vec1, size, 0.0f);
    utils::waitAndCheckForErrors();
}

__global__
void k_adam_kernel(float* m, float* v, float* w, const float* d, size_t max_size, float beta1, float beta2, float beta1_tpower, float beta2_tpower, float learning_rate)
{
    const float eps = 1e-8;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < max_size; i += blockDim.x * gridDim.x)
    {
        float d_temp = d[i];
        m[i] = m[i] * beta1 + d_temp * (1 - beta1);
        v[i] = v[i] * beta2 + d_temp * d_temp * (1 - beta2);
        float m_hat = m[i] / (1 - beta1_tpower);
        float v_hat = __fsqrt_rn(v[i] / (1 - beta2_tpower)) + eps;
        w[i] += (m_hat / v_hat) * (-learning_rate);
    }
}

__inline__ __device__ float4 operator+(const float4& a, const float4& b)
{
    float4 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    result.w = a.w + b.w;
    return result;
}

__inline__ __device__ float4 operator*(const float4& a, const float4& b)
{
    float4 result;
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    result.z = a.z * b.z;
    result.w = a.w * b.w;
    return result;
}

__inline__ __device__ float4 operator/(const float4& a, const float4& b)
{
    float4 result;
    result.x = a.x / b.x;
    result.y = a.y / b.y;
    result.z = a.z / b.z;
    result.w = a.w / b.w;
    return result;
}

__inline__ __device__ float4 sqrtf4(const float4& a)
{
    float4 result;
    result.x = __fsqrt_rn(a.x);
    result.y = __fsqrt_rn(a.y);
    result.z = __fsqrt_rn(a.z);
    result.w = __fsqrt_rn(a.w);
    return result;
}

__inline__ __device__ float4 operator+(const float4& a, const float& b)
{
    float4 result;
    result.x = a.x + b;
    result.y = a.y + b;
    result.z = a.z + b;
    result.w = a.w + b;
    return result;
}

__inline__ __device__ float4 operator*(const float4& a, const float& b)
{
    float4 result;
    result.x = a.x * b;
    result.y = a.y * b;
    result.z = a.z * b;
    result.w = a.w * b;
    return result;
}

__inline__ __device__ float4& operator+=(float4& a, const float4& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

__inline__ __device__ float4 operator/(const float4& a, const float& b)
{
    float4 result;
    result.x = a.x / b;
    result.y = a.y / b;
    result.z = a.z / b;
    result.w = a.w / b;
    return result;
}



__global__
void k_adam_kernel_vectorized(float* m, float* v, float* w, const float* d, size_t max_size, float beta1, float beta2, float beta1_tpower, float beta2_tpower, float learning_rate)
{
    const float eps = 1e-8;
    int i;
    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < max_size / 4; i += blockDim.x * gridDim.x)
    {
        float4 d_temp = reinterpret_cast<const float4*>(d)[i];
        reinterpret_cast<float4*>(m)[i] = (reinterpret_cast<float4*>(m)[i]) * beta1 + d_temp * (1 - beta1);
        reinterpret_cast<float4*>(v)[i] = (reinterpret_cast<float4*>(v)[i]) * beta2 + (d_temp * d_temp) * (1 - beta2);
        float4 m_hat = reinterpret_cast<float4*>(m)[i] / (1 - beta1_tpower);
        float4 v_hat = sqrtf4((reinterpret_cast<float4*>(v)[i]) / (1 - beta2_tpower)) + eps;
        reinterpret_cast<float4*>(w)[i] += (m_hat / v_hat) * (-learning_rate);
    }

    // in only one thread, process final elements (if there are any)
    int remainder = max_size % 4;
    if (i == max_size / 4 && remainder != 0) {
        while (remainder) {
            int idx = max_size - remainder--;
            float d_temp = d[idx];
            m[idx] = m[idx] * beta1 + d_temp * (1 - beta1);
            v[idx] = v[idx] * beta2 + d_temp * d_temp * (1 - beta2);
            float m_hat = m[idx] / (1 - beta1_tpower);
            float v_hat = __fsqrt_rn(v[idx] / (1 - beta2_tpower)) + eps;
            w[idx] += (m_hat / v_hat) * (-learning_rate);
        }
    }

}

void adam_kernel(float* m, float* v, float* w, const float* d, size_t max_size, float beta1, float beta2, float beta1_tpower, float beta2_tpower, float learning_rate)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    k_adam_kernel<< <32 * numSMs, 256 >> > (m, v, w, d, max_size, beta1, beta2, beta1_tpower, beta2_tpower, learning_rate);
    utils::waitAndCheckForErrors();
}

void adam_kernel_vectorized(float* m, float* v, float* w, const float* d, size_t max_size, float beta1, float beta2, float beta1_tpower, float beta2_tpower, float learning_rate)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    k_adam_kernel_vectorized << <32 * numSMs, 256 >> > (m, v, w, d, max_size, beta1, beta2, beta1_tpower, beta2_tpower, learning_rate);
    utils::waitAndCheckForErrors();
}