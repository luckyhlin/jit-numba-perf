import argparse

from numba import jit, cuda
import numpy as np
import time


def vec_add_vanilla(a, b, c):
    for i in range(len(a)):
        c[i] = a[i] + b[i]
    return c


@jit(nopython=True)
def vec_add_jit(a, b, c):
    for i in range(len(a)):
        c[i] = a[i] + b[i]
    return c


@cuda.jit()
def vec_add_kernel(a, b, result):
    idx = cuda.grid(1)
    if idx < a.size:
        result[idx] = a[idx] + b[idx]


def vec_add_cuda_jit(a, b, prompt):
    n = a.size
    a_device = cuda.to_device(a)
    b_device = cuda.to_device(b)
    result_device = cuda.device_array(n)

    threads_per_block = 1024
    blocks_per_grid = (a.size + (threads_per_block - 1)) // threads_per_block

    # Create CUDA events
    start_event = cuda.event()
    end_event = cuda.event()

    # Record the start event
    start_event.record()
    vec_add_kernel[blocks_per_grid, threads_per_block](a_device, b_device, result_device)
    # Record the end event and synchronize
    end_event.record()
    end_event.synchronize()

    elapsed_time = cuda.event_elapsed_time(start_event, end_event)
    print("Elapsed {} kernel time = {}ms".format(prompt, elapsed_time))

    result = result_device.copy_to_host()
    return result


def main():
    parser = argparse.ArgumentParser(description="Run Makefile commands.")
    parser.add_argument("N")
    parser.add_argument("P", help="whether to run vanilla python")
    args = parser.parse_args()

    n = int(args.N)
    p = int(args.P)

    vec_a = np.random.rand(n) * 100
    vec_b = np.random.rand(n) * 100
    vec_c = np.empty(n)
    # ---------------
    # @jit
    print('-' * 5)
    print('@jit')
    print('With compilation')
    start = time.perf_counter()
    result = vec_add_jit(vec_a, vec_b, vec_c)
    end = time.perf_counter()
    print("Elapsed jit with compilation = {}ms".format((end - start) * 1000))
    print(f"Sum of the result = {np.sum(result)}")

    print('After compilation')
    start = time.perf_counter()
    vec_add_jit(vec_a, vec_b, vec_c)
    end = time.perf_counter()
    print("Elapsed jit after compilation = {}ms".format((end - start) * 1000))

    # ---------------
    # @cuda.jit
    print('-' * 5)
    print('@cuda.jit')
    print('With compilation')
    start = time.perf_counter()
    vec_add_cuda_jit(vec_a, vec_b, 'with compilation')
    end = time.perf_counter()
    print("Elapsed total time = {}ms".format((end - start) * 1000))
    print(f"Sum of the result = {np.sum(result)}")

    print('After compilation')
    start = time.perf_counter()
    vec_add_cuda_jit(vec_a, vec_b, 'after compilation')
    end = time.perf_counter()
    print("Elapsed total time = {}ms".format((end - start) * 1000))

    # ---------------
    if p >= 1:
        print('-' * 5)
        print('vanilla python')
        print('First run')
        start = time.perf_counter()
        result = vec_add_vanilla(vec_a, vec_b, vec_c)
        end = time.perf_counter()
        print("Elapsed second run vanilla python = {}ms".format((end - start) * 1000))
        print(f"Sum of the result = {np.sum(result)}")

        print('Second run')
        start = time.perf_counter()
        vec_add_vanilla(vec_a, vec_b, vec_c)
        end = time.perf_counter()
        print("Elapsed first run vanilla python = {}ms".format((end - start) * 1000))


if __name__ == "__main__":
    main()

