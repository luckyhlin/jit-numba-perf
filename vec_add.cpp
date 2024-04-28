#include <iostream>
#include <chrono>
#include <random>

void vec_add_pointer(const float* a, const float* b, float* result, int size) {
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void vec_add_vector(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& result) {
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
}

int main(int argc, char* argv[]) {
    int N = 100 * 1000 * 1000; // default value
    N = std::atoi(argv[1]);

    // Allocate memory
    float* aPointer = new float[N];
    float* bPointer = new float[N];
    float* resultPointer = new float[N];

    std::vector<int> aVec(N), bVec(N), resultVec(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 100);

    for (int i = 0; i < N; ++i) {
        aPointer[i] = dis(gen);
        bPointer[i] = dis(gen);

        aVec[i] = aPointer[i];
        bVec[i] = bPointer[i];
    }

    auto start = std::chrono::high_resolution_clock::now();
    vec_add_pointer(aPointer, bPointer, resultPointer, N);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> diff = end - start;
    std::cout << "Elapsed pointer time: " << diff.count() * 1000 << "ms\n";

    std::cout << "-----" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    vec_add_vector(aVec, bVec, resultVec);
    end = std::chrono::high_resolution_clock::now();

    diff = end - start;
    std::cout << "Elapsed vector time: " << diff.count() * 1000 << "ms\n";

    // Deallocate memory
    delete[] aPointer;
    delete[] bPointer;
    delete[] resultPointer;

    return 0;
}
