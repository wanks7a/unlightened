#include <shape.h>

void sum_all_values(const shape& sh, const float* input, float* value);
//performs vec1/vec2 element wise
void vector_divide(float* vec1, float* vec2, size_t size);
void vector_scale(float* vec1, size_t size, float value);
void vector_add(float* vec1, size_t size, float value);
void vector_mul(float* vec1, const float* vec2, size_t size);
void vector_add(float* vec1, const float* vec2, size_t size);
void vector_sqrt(float* vec1, size_t size);
void adam_kernel(float* m, float* v, float* w, const float* d, size_t max_size, float beta1, float beta2, float beta1_tpower, float beta2_tpower, float learning_rate);
void adam_kernel_vectorized(float* m, float* v, float* w, const float* d, size_t max_size, float beta1, float beta2, float beta1_tpower, float beta2_tpower, float learning_rate);