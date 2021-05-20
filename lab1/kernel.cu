
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <iostream>

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

int main(int argc, char* argv[]) {
	double const minValue = -10;
	double const maxValue = 10;

	size_t const N = 4;

	double* A = fillRandomMatrix(N, minValue, maxValue);
	double* B = fillRandomMatrix(N, minValue, maxValue);
	
	delete[] A;
	delete[] B;
}
