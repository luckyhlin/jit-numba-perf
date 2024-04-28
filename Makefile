vec_add:
	g++ -Wall -O3 -std=c++17 -o vec_add vec_add.cpp
	./vec_add $(N)

vec_add_cuda:
	nvcc -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o vec_add_cuda vec_add.cu
	./vec_add_cuda $(N)

vec_add_thrust:
	nvcc -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o vec_add_thrust vec_add_thrust.cu
	./vec_add_thrust $(N)

vec_add_ptx:
	llc-17 -march=nvptx64 -mcpu=sm_86 vec_add_kernel_ir.ll -o vec_add_kernel.ptx
	nvcc vec_add_ptx_framework.cu vec_add_ptx_kernel.cu -o vec_add_ptx -lcuda
	./vec_add_ptx $(N)

clean:
	-rm vec_add vec_add_cuda vec_add_thrust vec_add_ptx