#pragma once
#include <cuda_runtime.h>

__global__ void convolveArrayHorizontal(float* input, float* output, size_t dataPitch, int XX, int YY, float scale);

__global__ void convolveArrayVertical(float* input, float* output, size_t dataPitch, int XX, int YY, float scale);

__global__ void add(float* a, float* b, float* out, float sa, float sb, size_t XX, size_t YY);

__global__ void mult(float* a, float* b, float* out, float scale, size_t XX, size_t YY);

__global__ void applyBoundary(float* positions, float* boundary, float* out, float amount, size_t XX, size_t YY);

__global__ void gaussianAdd(float2 position, float2 boundary, float size, float mult, float* out, size_t width, size_t height, size_t pitch);

__global__ void writeBack(float* d_position, float* deviceResults, float2 destPosition, float2 boundary, size_t iteration, size_t width, size_t height);