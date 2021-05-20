
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <windows.h>

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

double processMultiplyMatrixOnCPU(double* A, double* B, double* res, size_t N) {
	LARGE_INTEGER start, end, freq;
	QueryPerformanceFrequency(&freq);

	QueryPerformanceCounter(&start);
	multiplyMatrixOnCPU(A, B, res, N);
	QueryPerformanceCounter(&end);
	
	return static_cast<double>(end.QuadPart - start.QuadPart) / freq.QuadPart;
}

int main(int argc, char* argv[]) {
	double const minValue = -10;
	double const maxValue = 10;

	size_t const N = 100;

	double* A = fillRandomMatrix(N, minValue, maxValue);
	double* B = fillRandomMatrix(N, minValue, maxValue);
	
	double* cpuResMatrix = new double[N * N];
	double cpuResTime = processMultiplyMatrixOnCPU(A, B, cpuResMatrix, N);

	std::cout << "Matrix multiplication time on CPU: " << cpuResTime << std::endl;

	delete[] A;
	delete[] B;

	delete[] cpuResMatrix;
}
