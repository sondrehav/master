// Surfaces for writing
#include <cstdio>
#include <driver_types.h>
#include <texture_types.h>
#include <vector_functions.hpp>
#include "../helper_math.h"

// Surfaces for writing
surface<void, 2> velocitySurfRef;
surface<void, 2> pressureSurfRef;
surface<void, 2> geometrySurfRef;
surface<void, 2> forcesSurfRef;

// Textures for reading
texture<float, 2, cudaReadModeElementType> velocityTexRef;
texture<float, 2, cudaReadModeElementType> pressureTexRef;
texture<float, 2, cudaReadModeElementType> forcesTexRef;
texture<float, 2, cudaReadModeElementType> geometryTexRef;

__constant__ float soundVelocity;
__constant__ float stepSize;
__constant__ int width;
__constant__ int height;
__constant__ int pmlLayers;
__constant__ float pmlMax;

__constant__ size_t numInputSamples;
__constant__ size_t numOutputSamples;

__constant__ int numInputChannels;
__constant__ int numOutputChannels;

__constant__ int2 inputLocations[2];
__constant__ int2 outputLocations[2];

#define LAPLACIAN {2.0f, -27.0f, 270.0f, -490.0f, 270.0f, -27.0f, 2.0f}
#define LAPLACIAN_N 7
#define LAPLACIAN_MULT 180.0f

extern "C" float __device__ laplacian(float2 location)
{
	const float conv[LAPLACIAN_N] = LAPLACIAN;
	float value = 0;
	for (int i = -LAPLACIAN_N / 2; i <= LAPLACIAN_N / 2; i++)
	{
		float values = tex2D(pressureTexRef, location.x + i, location.y);
		value += values * conv[i + LAPLACIAN_N / 2];
	}
	for (int i = -LAPLACIAN_N / 2; i <= LAPLACIAN_N / 2; i++)
	{
		float values = tex2D(pressureTexRef, location.x, location.y + i);
		value += values * conv[i + LAPLACIAN_N / 2];
	}
	return value / LAPLACIAN_MULT;
}

__device__ __inline__ float getPMLDampingValue(int x, int y)
{
	float value = 0.0f;

	if (x < pmlLayers) value += 1.0 - (float)x / pmlLayers;
	else if (x >= width - pmlLayers) value += (float)(x - width + pmlLayers) / pmlLayers;

	if (y < pmlLayers) value += 1.0 - (float)y / pmlLayers;
	else if (y >= height - pmlLayers) value += (float)(y - height + pmlLayers) / pmlLayers;

	return value * pmlMax;
}

extern "C" __global__ void iterateVelocity(float timeStep, float* input, int sample, float wallAbsorbtion)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		
		float2 location = make_float2((float)x + 0.5, (float)y + 0.5);

		float forces = 0;
		if(sample < numInputSamples) forces = tex2D(forcesTexRef, location.x - pmlLayers, location.y - pmlLayers) * input[sample];

		float currentVelocity = tex2D(velocityTexRef, location.x, location.y);
		float walls = fminf(tex2D(geometryTexRef, location.x - pmlLayers, location.y - pmlLayers) / fmaxf(wallAbsorbtion, 0.000001), 1.0);

		float dVelocity = timeStep * (forces + soundVelocity * soundVelocity * laplacian(location) / (stepSize * stepSize));
		float pmlValue = fmaxf(1.0 - getPMLDampingValue(x, y), 0);

		surf2Dwrite<float>((1.0 - walls) * (currentVelocity + dVelocity) * pmlValue, velocitySurfRef, x * sizeof(float), y);
	}
}

extern "C" __global__ void iteratePressure(float timeStep)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		float2 location = make_float2((float)x + 0.5, (float)y + 0.5);
		float currentPressure = tex2D(pressureTexRef, location.x, location.y);
		float velocity = tex2D(velocityTexRef, location.x, location.y);

		float dPressure = timeStep * velocity;
		float pmlValue = fmaxf(1.0 - getPMLDampingValue(x, y), 0);

		//surf2Dwrite<float>(currentPressure + pmlValue * dPressure, pressureSurfRef, x * sizeof(float), y);
		surf2Dwrite<float>(currentPressure + pmlValue * dPressure, pressureSurfRef, x * sizeof(float), y);

	}
}

extern "C" __global__ void firstIterationVelocity(float timeStep)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		float2 location = make_float2((float)x + 0.5, (float)y + 0.5);
		float currentVelocity = tex2D(velocityTexRef, location.x, location.y);
		//float forces = tex2D(forcesTexRef, location.x, location.y);

		float dVelocity = 0.5 * timeStep * (soundVelocity * soundVelocity * laplacian(location) / (stepSize * stepSize));

		float pmlValue = fmaxf(1.0 - getPMLDampingValue(x, y), 0);

		surf2Dwrite<float>((currentVelocity + dVelocity) * pmlValue, velocitySurfRef, x * sizeof(float), y);
	}
}


/**/

float __inline__ __device__ amplitude(float distanceFromCenter, float brushSize, float brushFalloff)
{
	return fminf(1.0, (brushSize - distanceFromCenter) / brushFalloff);
}

extern "C" __global__ void drawLine(int2 from, int2 to, float brushSize, float brushFalloff)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width - 2 * pmlLayers && y < height - 2 * pmlLayers)
	{
		
		float2 location = make_float2(x, y);
		float2 readLocation = location + 0.5f;

		float2 ffrom = make_float2(from);
		float2 fto = make_float2(to);

		float2 projection = fto - ffrom;
		float2 tempLocationA = location - ffrom;
		float2 tempLocationB = location - fto;

		float2 projectedLocation = projection * dot(tempLocationA, projection) / dot(projection, projection) + ffrom;

		float value = 0;
		float dist;

		if (dot(projection, tempLocationA) < 0)
		{
			if((dist = length(tempLocationA)) < brushSize)
			{
				value = amplitude(dist, brushSize, brushFalloff);
			}
		} else if (dot(projection, tempLocationB) > 0)
		{
			if ((dist = length(tempLocationB)) < brushSize)
			{
				value = amplitude(dist, brushSize, brushFalloff);
			}
		} else if((dist = length(projectedLocation - location)) < brushSize)
		{
			value = amplitude(dist, brushSize, brushFalloff);
		}

		float newValue = tex2D(geometryTexRef, readLocation.x, readLocation.y);
		newValue = fmaxf(fminf(value + newValue, 1.0), 0.0);
		
		surf2Dwrite<float>(newValue, geometrySurfRef, x * sizeof(float), y);
	}
}

extern "C" __global__ void sampleAt(float* output, int sample)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < numOutputChannels)
	{
		int2 location = outputLocations[x];
		float currentPressure = tex2D(pressureTexRef, location.x + 0.5 + pmlLayers, location.y + 0.5 + pmlLayers);
		output[numOutputSamples * x + sample] = currentPressure;
	}
}

extern "C" __global__ void sampleIn(float* input, int sample, float amp)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < numInputChannels)
	{
		int2 location = inputLocations[x];
		float value = input[numInputSamples * x + sample] * amp;
		//printf("%3.3f, %d, %d\n", value, sample, (int)numInputSamples);
		surf2Dwrite<float>(value, forcesSurfRef, location.x * sizeof(float), location.y);
	}
}
