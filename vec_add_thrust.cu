#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <random>
#include <chrono>
#include <iostream>

int main() {
    const int N = 100 * 1000 * 1000;
    thrust::host_vector<float> hA(N), hB(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 100);

    for (int i = 0; i < N; i ++) {
        hA[i] = dis(gen);
        hB[i] = dis(gen);
    }

    thrust::device_vector<float> dA = hA;
    thrust::device_vector<float> dB = hB;
    thrust::device_vector<float> dResult(N);

    auto start = std::chrono::high_resolution_clock::now();

    // Perform vector addition
    thrust::transform(dA.begin(), dA.end(), dB.begin(), dResult.begin(), thrust::plus<float>());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Elapsed time = " << duration.count() << "ms" << std::endl;

    return 0;
}
