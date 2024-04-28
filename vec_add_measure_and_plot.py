import numpy as np
import time
from statistics import median
import subprocess
import re
import matplotlib.pyplot as plt

trials = 5


def measure_time_cpp(command):
    elapsed_times_pointer = []
    elapsed_times_vector = []

    for i in range(trials):
        result = subprocess.run([command], shell=True, capture_output=True, text=True)
        lines = result.stdout.split('\n')

        found_flag = False
        for line in lines:
            if "Elapsed" in line:
                match = re.search(r"(\d+\.\d+)", line)
                if match:
                    elapsed_time = float(match.group(1))
                    if "pointer" in line:
                        elapsed_times_pointer.append(elapsed_time)
                    else:
                        elapsed_times_vector.append(elapsed_time)
                    found_flag = True

        if not found_flag:
            raise ValueError("Time not found in the output")

    return median(elapsed_times_pointer), median(elapsed_times_vector)

def measure_time_cuda(command):
    elapsed_times = []

    for i in range(trials):
        result = subprocess.run([command], shell=True, capture_output=True, text=True)
        lines = result.stdout.split('\n')

        found_flag = False
        for line in lines:
            if "Elapsed" in line:
                match = re.search(r"(\d+\.\d+)", line)
                if match:
                    elapsed_times.append(float(match.group(1)))
                    found_flag = True
                    break
        if not found_flag:
            raise ValueError("Time not found in the output")

    return median(elapsed_times)


# Function to measure execution time for Python and Numba methods
def measure_time_python(command):
    elapsed_times_jit = []
    elapsed_times_cuda_jit = []

    for i in range(trials):
        result = subprocess.run([command], shell=True, capture_output=True, text=True)
        lines = result.stdout.split('\n')

        found_flag = False
        for line in lines:
            if "Elapsed" in line:
                match = re.search(r"(\d+\.\d+)", line)
                if match:
                    elapsed_time = float(match.group(1))
                    if "jit after compilation" in line:
                        elapsed_times_jit.append(elapsed_time)
                    if "after compilation":
                        elapsed_times_cuda_jit.append(elapsed_time)
                    found_flag = True

        if not found_flag:
            raise ValueError("Time not found in the output")

    return median(elapsed_times_jit), median(elapsed_times_cuda_jit)

# Define a range of sizes for N
sizes = [2 ** i for i in range(10, 26, 2)]
# times_python = []
times_numba_jit = []
times_numba_cuda_jit = []
times_cpp_pointer = []
times_cpp_vector = []
times_cuda = []
times_ptx = []

# Measure time for each size and method
for N in sizes:
    numba_jit, numba_cuda_jit = measure_time_python(f'python vec_add.py {N} 0')
    times_numba_jit.append(numba_jit)
    times_numba_cuda_jit.append(numba_cuda_jit)
    pointer_time, vector_time = measure_time_cpp(f'make clean && make vec_add N={N}')
    times_cpp_pointer.append(pointer_time)
    times_cpp_vector.append(vector_time)
    times_cuda.append(measure_time_cuda(f'make clean && make vec_add_cuda N={N}'))
    times_ptx.append(measure_time_cuda(f'make clean && make vec_add_ptx N={N}'))

# Plotting the results
plt.plot(sizes, times_numba_jit, marker='o', label='Numba @jit')
plt.plot(sizes, times_numba_cuda_jit, marker='o', label='Numba @cuda.jit')
plt.plot(sizes, times_cpp_pointer, marker='o', label='Vanilla C++ with pointer')
plt.plot(sizes, times_cpp_vector, marker='o', label='Vanilla C++ with vector')
plt.plot(sizes, times_cuda, marker='o', label='CUDA')
plt.plot(sizes, times_ptx, marker='o', label='NVPTX')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Array Size N - log scale')
plt.ylabel('Execution Time (milliseconds) - log scale')
plt.title('Vector Addition For Various Implementations: Execution Time vs Size')
plt.legend()
plt.grid(True)
plt.show()
