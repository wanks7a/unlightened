#include <GpuUtils.h>
#include <iostream>
#include <cmath>

namespace utils
{
    unsigned int getBlockSize(size_t threadsPerBlock, size_t maxThreads)
    {
        return static_cast<unsigned int>(std::ceil(static_cast<double>(maxThreads) / threadsPerBlock));
    }

    bool GpuInit()
    {
        // Choose which GPU to run on, change this on a multi-GPU system.
        auto cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!");
            return false;
        }
        return true;
    }

    bool GpuRelase()
    {
        auto cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return false;
        }
        return true;
    }

    void waitAndCheckForErrors()
    {
        // Check for any errors launching the kernel
        auto cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        }
    }
}