
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <windows.h>

static const size_t BLOCK_SIZE = 32;
static const size_t MATRIX_N[4] = { 500, 1000, 1500, 2000 };

double* fillRandomMatrix(size_t const N, double const minValue, double const maxValue) {
	size_t length = N * N;
	double* res = new double[length];

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distr(minValue, maxValue);

	for (size_t i = 0; i < length; ++i) {
		res[i] = distr(gen);
	}

	return res;
}

void multiplyMatrixOnCPU(double* A, double* B, double* res, size_t N) {
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < N; ++j) {
			res[i * N + j] = 0;

			for (size_t k = 0; k < N; ++k) {
				res[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}

__global__ void multiplyMatrixKernel(double* A, double* B, double* res, size_t N) {
	size_t const i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t const j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= N || j >= N) {
		return;
	}

	res[i * N + j] = 0;
	for (size_t k = 0; k < N; ++k) {
		res[i * N + j] += A[i * N + k] * B[k * N + j];
	}
}

double processMultiplyMatrixOnCPU(double* A, double* B, double* res, size_t N) {
	LARGE_INTEGER start, end, freq;
	QueryPerformanceFrequency(&freq);

	QueryPerformanceCounter(&start);
	multiplyMatrixOnCPU(A, B, res, N);
	QueryPerformanceCounter(&end);
	
	return static_cast<double>(end.QuadPart - start.QuadPart) / freq.QuadPart;
}

double processMultiplyMatrixOnGPU(double* A, double* B, double* res, size_t N) {
	float gpuTime = 0.0f;
	size_t size = N * N * sizeof(double);
	cudaEvent_t start, end;

	//allocate device memory
	double* ADev = NULL;
	double* BDev = NULL;
	double* resDev = NULL;

	cudaMalloc((void**)&ADev, size);
	cudaMalloc((void**)&BDev, size);
	cudaMalloc((void**)&resDev, size);

	//set kernel launch configuration
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	//create cuda event handles
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);

	cudaMemcpy(ADev, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(BDev, B, size, cudaMemcpyHostToDevice);

	multiplyMatrixKernel<<<blocks, threads>>> (ADev, BDev, resDev, N);

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpuTime, start, end);

	cudaMemcpy(res, resDev, size, cudaMemcpyDeviceToHost);

	//release resources
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	return gpuTime / 1000.0f;
}


int main(int argc, char* argv[]) {
	double const minValue = -10;
	double const maxValue = 10;

	for (size_t N : MATRIX_N) {
		double* A = fillRandomMatrix(N, minValue, maxValue);
		double* B = fillRandomMatrix(N, minValue, maxValue);

		double* cpuResMatrix = new double[N * N];
		double* gpuResMatrix = new double[N * N];
		double cpuResTime = processMultiplyMatrixOnCPU(A, B, cpuResMatrix, N);
		double gpuResTime = processMultiplyMatrixOnGPU(A, B, gpuResMatrix, N);

		std::cout << "-------------------------------------------------" << std::endl;
		std::cout << "Matrix size N = " << N << std::endl;
		std::cout << "Matrix multiplication time on CPU: " << cpuResTime << std::endl;
		std::cout << "Matrix multiplication time on GPU: " << gpuResTime << std::endl;

		delete[] A;
		delete[] B;

		delete[] cpuResMatrix;
		delete[] gpuResMatrix;
	}
}
