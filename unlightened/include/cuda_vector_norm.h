#pragma once

float cuda_vector_norm(float* ptr, size_t size);
bool  cuda_scale_vector(float* ptr, size_t size, float scale);