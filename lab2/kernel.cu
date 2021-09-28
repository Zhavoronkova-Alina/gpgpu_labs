/*
* Calculate matrix multiplication on CPU and GPU
* @file kernel.cu
* @author Alina Zhavoronkova
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <windows.h>

static const size_t BLOCK_SIZE = 32;
static const size_t MATRIX_N[4] = { 500, 1000, 1500 };

/*
* Fill matrix with uniform distributed random numbers from minValue to maxValue
* @param N - Size of matrix (square matrix NxN)
* @param minValue - The lower bound of the numbers
* @param maxValue - The upper bound of the numbers
* @return The filled matrix
*/
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

/*
* Multiply matrices using CPU
* @param A - The first matrix
* @param B - The second matrix
* @param res - The matrix, where result of multiplication is written
* @param N - Size of matrices (square matrix NxN)
*/
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

/*
* Kernel used for matrix multiplication using GPU
* @param A - The first matrix
* @param B - The second matrix
* @param res - The matrix, where result of multiplication is written
* @param N - Size of matrices (square matrix NxN)
*/
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

__global__ void multiplyMatrixSharedKernel(double* A, double* B, double* res, size_t N) {
	size_t ty = threadIdx.y;
	size_t tx = threadIdx.x;

	size_t bx = blockIdx.x;
	size_t by = blockIdx.y;

	size_t const i = blockDim.y * by + ty;
	size_t const j = blockDim.x * bx + tx;

	double sum = 0.0;

	for (size_t k = 0; k * BLOCK_SIZE < N; ++k) {
		__shared__ double subA[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ double subB[BLOCK_SIZE][BLOCK_SIZE];

		size_t const aj = tx + k * BLOCK_SIZE;
		size_t const bi = ty + k * BLOCK_SIZE;

		subA[ty][tx] = 0;
		subB[ty][tx] = 0;

		//select submatrices (BLOCK_SIZE x BLOCK_SIZE) from A and from B
		if (i < N && aj < N) {
			subA[ty][tx] = A[i * BLOCK_SIZE + aj];
		}

		if (j < N && bi < N) {
			subB[ty][tx] = B[bi * BLOCK_SIZE + j];
		}

		__syncthreads(); // to make sure the matrices are loaded

		for (size_t s = 0; s < BLOCK_SIZE; ++k) {
			sum += subA[ty][s] * subB[s][tx];
		}

		__syncthreads(); // to make sure submatrices not needed		
	}

	if (i < N && j < N) {
		res[i * N + j] = sum;
	}
}

/*
* Multiplies matrices on CPU and counts the execution time
* @param A - The first matrix
* @param B - The second matrix
* @param res - The matrix, where result of multiplication is written
* @param N - Size of matrices (square matrix NxN)
* @return Time spent on matrix multiplication
*/
double processMultiplyMatrixOnCPU(double* A, double* B, double* res, size_t N) {
	LARGE_INTEGER start, end, freq;
	QueryPerformanceFrequency(&freq);

	QueryPerformanceCounter(&start);
	multiplyMatrixOnCPU(A, B, res, N);
	QueryPerformanceCounter(&end);

	return static_cast<double>(end.QuadPart - start.QuadPart) / freq.QuadPart;
}

/*
* Multiplies matrices on GPU and counts the execution time
* @param A - The first matrix
* @param B - The second matrix
* @param res - The matrix, where result of multiplication is written
* @param N - Size of matrices (square matrix NxN)
* @return Time spent on matrix multiplication
*/
double processMultiplyMatrixOnGPU(double* A, double* B, double* res, size_t N, bool shared_mem) {
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
	dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

	//create cuda event handles
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);

	cudaMemcpy(ADev, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(BDev, B, size, cudaMemcpyHostToDevice);

	if (shared_mem) {
		multiplyMatrixSharedKernel<<<blocks, threads>>> (ADev, BDev, resDev, N);
	}
	else {
		multiplyMatrixKernel <<<blocks, threads>>> (ADev, BDev, resDev, N);
	}

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpuTime, start, end);

	cudaMemcpy(res, resDev, size, cudaMemcpyDeviceToHost);

	//release resources
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	cudaFree(ADev);
	cudaFree(BDev);
	cudaFree(resDev);

	return gpuTime / 1000.0f;
}

/*
* Calculate the maximum difference between the corresponding elements of two matrices
* @param A - The first matrix
* @param B - The second matrix
* @param N - Size of matrices (square matrix NxN)
* @return Maximum difference between the corresponding elements of two matrices
*/
double getMaxMatrixDifference(double* A, double* B, size_t N) {
	double res = 0;

	for (size_t i = 0; i < N * N; ++i)
		res = std::max(res, std::fabs(A[i] - B[i]));

	return res;
}

int main(int argc, char* argv[]) {
	double const minValue = -10;
	double const maxValue = 10;

	for (size_t N : MATRIX_N) {
		double* A = fillRandomMatrix(N, minValue, maxValue);
		double* B = fillRandomMatrix(N, minValue, maxValue);

		double* cpuResMatrix = new double[N * N];
		double* gpuResMatrix = new double[N * N];
		double* gpuSharedResMatrix = new double[N * N];

		double cpuResTime = processMultiplyMatrixOnCPU(A, B, cpuResMatrix, N);
		double gpuResTime = processMultiplyMatrixOnGPU(A, B, gpuResMatrix, N, false);
		double gpuSharedResTime = processMultiplyMatrixOnGPU(A, B, gpuSharedResMatrix, N, true);

		double maxDiff = getMaxMatrixDifference(gpuResMatrix, gpuSharedResMatrix, N);

		std::cout << "-------------------------------------------------" << std::endl;
		std::cout << "Matrix size N = " << N << std::endl;
		std::cout << "Matrix multiplication time on CPU:        " << cpuResTime << std::endl;
		std::cout << "Matrix multiplication time on GPU:        " << gpuResTime << std::endl;
		std::cout << "Matrix Shared multiplication time on GPU: " << gpuSharedResTime << std::endl;
		std::cout << "Maximum matrix difference: " << maxDiff << std::endl;

		delete[] A;
		delete[] B;

		delete[] cpuResMatrix;
		delete[] gpuResMatrix;
		delete[] gpuSharedResMatrix;
	}
}