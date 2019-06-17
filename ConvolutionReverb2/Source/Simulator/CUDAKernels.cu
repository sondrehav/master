#include "CUDAKernels.h"
#include <cstdio>

const size_t convolutionSize = 7;
__constant__ float convolutionKernel[convolutionSize]{ 2, -27, 270, -490, 270, -27, 2 };

__global__ void convolveArrayHorizontal(float* input, float* output, size_t dataPitch, int XX, int YY, float scale)
{
	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;
	if ((tidx < XX) && (tidy < YY))
	{
		float* row = (float *)((char*)output + tidy * dataPitch);
		float* in = (float *)((char*)input + tidy * dataPitch);

		float value = 0.0;
		for (int i = 0; i < convolutionSize; i++)
		{
			int ax = tidx + i - convolutionSize / 2;
			value += ax >= 0 ? (ax < XX ? convolutionKernel[i] * in[ax] : 0.0f) : 0.0f;
		}
		row[tidx] = value * scale;
	}
}

__global__ void convolveArrayVertical(float* input, float* output, size_t dataPitch, int XX, int YY, float scale)
{
	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;
	if ((tidx < XX) && (tidy < YY))
	{
		float* row = (float *)((char*)output + tidy * dataPitch);
		float* in = (float *)((char*)input + tidy * dataPitch);
		float value = 0.0;
		for (int i = 0; i < convolutionSize; i++)
		{
			int ay = tidx + (i - convolutionSize / 2) * (dataPitch / sizeof(float));
			int yy = (sizeof(float)) * ay / dataPitch + tidy;
			value += yy >= 0 ? (yy < YY ? convolutionKernel[i] * in[ay] : 0.0f) : 0.0f;
		}
		row[tidx] = value * scale;
	}
}

__global__ void add(float* a, float* b, float* out, float sa, float sb, size_t XX, size_t YY)
{
	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;
	if ((tidx < XX) && (tidy < YY))
	{
		out[tidy*XX + tidx] = a[tidy*XX + tidx] * sa + b[tidy*XX + tidx] * sb;
	}
}

__global__ void mult(float* a, float* b, float* out, float scale, size_t XX, size_t YY)
{
	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;
	if ((tidx < XX) && (tidy < YY))
	{
		out[tidy*XX + tidx] = a[tidy*XX + tidx] * b[tidy*XX + tidx] * scale;
	}
}

__global__ void applyBoundary(float* positions, float* boundary, float* out, float amount, size_t XX, size_t YY)
{
	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;
	if ((tidx < XX) && (tidy < YY))
	{
		out[tidy*XX + tidx] = positions[tidy*XX + tidx] * (boundary[tidy*XX + tidx] * amount + (1.0f - amount));
	}
}

__global__ void gaussianAdd(float2 position, float2 boundary, float size, float mult, float* out, size_t width, size_t height, size_t pitch)
{
	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;
	if ((tidx < width) && (tidy < height))
	{
		// np.exp(-pow(x - position[0], 2) / (2 * size * size) - pow(y - position[1], 2) / (2 * size * size))
		float x = boundary.x * (float) tidx / width;
		float y = boundary.y * (float) tidy / height;
		float n = 2 * size * size;
		out[tidy*(pitch / sizeof(float)) + tidx] += mult * expf( - powf(x - position.x, 2) / n - powf(y - position.y, 2) / n);
	}
}


__global__ void writeBack(float* d_position, float* deviceResults, float2 destPosition, float2 boundary, size_t iteration, size_t width, size_t height)
{
	float ix = destPosition.x / boundary.x;
	float iy = destPosition.y / boundary.y;
	size_t indexX = floor(ix * width);
	size_t indexY = floor(iy * height);
	deviceResults[iteration] = d_position[width * indexY + indexX];
}