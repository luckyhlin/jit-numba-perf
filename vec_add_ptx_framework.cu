#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <cuda_runtime.h>

// Function prototype
cudaError_t launchKernel(const char* ptxCode, float* A, float* B, float* C, int N);

int main(int argc, char* argv[]) {
    int N = 100 * 1000 * 1000; // default value
    N = std::atoi(argv[1]);
    int size = N * sizeof(float);
    float *A, *B, *C;
    float *dA, *dB, *dC; // Device pointers

    // Allocate host memory
    A = new float[N];
    B = new float[N];
    C = new float[N];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 100);

    for (int i = 0; i < N; i ++) {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }

    // Allocate device memory
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    // Copy data from host to device
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    // Load the PTX code from file
    std::ifstream ptxFileStream("vec_add_kernel.ptx");
    if (!ptxFileStream.is_open()) {
        std::cerr << "Failed to open PTX file." << std::endl;
        return 1;
    }
    std::string ptxCode((std::istreambuf_iterator<char>(ptxFileStream)), std::istreambuf_iterator<char>());
    if (ptxCode.empty()) {
        std::cerr << "PTX code is empty." << std::endl;
        return 1;
    }



    // Launch the kernel
    cudaError_t cudaStatus = launchKernel(ptxCode.c_str(), dA, dB, dC, N);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);


    // Cleanup
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
