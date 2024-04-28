#include <iostream>
#include <chrono>
#include <random>
#include <cstdlib>

__global__ void vec_add_kernel(float *a, float *b, float *result, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        result[index] = a[index] + b[index];
    }
}

int main(int argc, char* argv[]) {
    int N = 100 * 1000 * 1000; // default value
    N = std::atoi(argv[1]);
    int size = N * sizeof(float);

    float *hA = new float[N];
    float *hB = new float[N];
    float *dA, *dB, *dResult;
    cudaMalloc((void **)&dA, size);
    cudaMalloc((void **)&dB, size);
    cudaMalloc((void **)&dResult, size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 100);

    for (int i = 0; i < N; i ++) {
        hA[i] = dis(gen);
        hB[i] = dis(gen);
    }

    // move data to device
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vec_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dResult, N);
    cudaEventRecord(stop);

    // move result back to host
    cudaMemcpy(hB, dB, size, cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed CUDA kernal time: " << milliseconds << "ms\n";

    // deallocate memory
    delete[] hA;
    delete[] hB;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dResult);

    return 0;
}
