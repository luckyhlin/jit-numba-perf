#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <iostream>
#include <string>

cudaError_t launchKernel(const char* ptxCode, float* d_A, float* d_B, float* d_C, int N) {
    CUmodule cuModule;
    CUfunction cuFunction;
    CUresult res;

    // Initialize CUDA Driver API
    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuInit failed: res=" << res << std::endl;
        return cudaErrorInitializationError;
    }

    // Get a handle to the first CUDA device
    CUdevice cuDevice;
    res = cuDeviceGet(&cuDevice, 0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuDeviceGet failed: res=" << res << std::endl;
        return cudaErrorInitializationError;
    }

    // Create a context
    CUcontext cuContext;
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuCtxCreate failed: res=" << res << std::endl;
        return cudaErrorInitializationError;
    }

    // Load PTX from string
    res = cuModuleLoadData(&cuModule, ptxCode);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuModuleLoadData failed: res=" << res << std::endl;
        cuCtxDestroy(cuContext);
        return cudaErrorInitializationError;
    }

    // Get a handle to the kernel function
    res = cuModuleGetFunction(&cuFunction, cuModule, "vectorAdd");
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuModuleGetFunction failed: res=" << res << std::endl;
        cuModuleUnload(cuModule);
        cuCtxDestroy(cuContext);
        return cudaErrorInitializationError;
    }

    // Set kernel parameters
    void* args[] = { &d_A, &d_B, &d_C, &N };

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    cudaEventRecord(start);
    res = cuLaunchKernel(cuFunction, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, NULL, args, NULL);
    cudaEventRecord(stop);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuLaunchKernel failed: res=" << res << std::endl;
        cuModuleUnload(cuModule);
        cuCtxDestroy(cuContext);
        return cudaErrorLaunchFailure;
    }

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed CUDA kernal time: " << milliseconds << "ms\n";

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cuModuleUnload(cuModule);
    cuCtxDestroy(cuContext);

    return cudaSuccess;
}
