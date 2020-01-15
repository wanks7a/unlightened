#pragma once
#ifndef _GPU_UTILS_H
#define _GPU_UTILS_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

namespace utils
{
size_t getBlockSize(size_t threadsPerBlock, size_t maxThreads);
bool GpuInit();
void waitAndCheckForErrors();
bool GpuRelase();
}
#endif
