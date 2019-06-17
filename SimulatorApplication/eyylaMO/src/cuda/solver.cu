// Surfaces for writing
#include <cstdio>
#include <driver_types.h>
#include <texture_types.h>
#include <vector_functions.hpp>
#include "../helper_math.h"

// Surfaces for writing
surface<void, 2> velocityXSurfRef;
surface<void, 2> velocityYSurfRef;
surface<void, 2> pressureSurfRef;
surface<void, 2> geometrySurfRef;
surface<void, 2> forcesSurfRef;
surface<void, 2> psiSurfRef;
surface<void, 2> phiSurfRef;

// Textures for reading
texture<float, 2, cudaReadModeElementType> velocityXTexRef;
texture<float, 2, cudaReadModeElementType> velocityYTexRef;
texture<float, 2, cudaReadModeElementType> pressureTexRef;
texture<float, 2, cudaReadModeElementType> forcesTexRef;
texture<float, 2, cudaReadModeElementType> geometryTexRef;
texture<float, 2, cudaReadModeElementType> psiTexRef;
texture<float, 2, cudaReadModeElementType> phiTexRef;

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

#define LAPLACIAN {1.0f, -13.88888889f, 250.0f, -250.0f, 13.88888889f, -1.0f}
#define LAPLACIAN_N 6
#define LAPLACIAN_MULT 213.333333333333f

extern "C" float2 __device__ gradient(float2 location, texture<float, 2, cudaReadModeElementType> tex)
{
	float p = tex2D(tex, location.x + 0.5, location.y + 0.5);
	float vx = p - tex2D(tex, location.x + 0.5 - 1.0, location.y + 0.5);
	float vy = p - tex2D(tex, location.x + 0.5, location.y + 0.5 - 1.0);
	return make_float2(vx, vy);
}

extern "C" float __device__ divergence_x(float2 location, texture<float, 2, cudaReadModeElementType> tex_x)
{
	return tex2D(tex_x, location.x + 0.5 + 1, location.y + 0.5) - tex2D(tex_x, location.x + 0.5, location.y + 0.5);
}


extern "C" float __device__ divergence_y(float2 location, texture<float, 2, cudaReadModeElementType> tex_y)
{
	return tex2D(tex_y, location.x + 0.5, location.y + 0.5 + 1) - tex2D(tex_y, location.x + 0.5, location.y + 0.5);
}

extern "C" float __device__ divergence(float2 location, texture<float, 2, cudaReadModeElementType> tex_x, texture<float, 2, cudaReadModeElementType> tex_y)
{
	return divergence_x(location, tex_x) + divergence_y(location, tex_y);
}

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

extern "C" __global__ void iterateVelocity(float timeStep)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		
		float2 loc = make_float2(x, y);
		float2 grad = gradient(loc, pressureTexRef) + gradient(loc, forcesTexRef);
		
		float2 pml = make_float2(
			fminf(fmaxf(fmaxf(0.0f, 1.0f - (loc.x - 0.5f) / (float)pmlLayers), (loc.x - 0.5f + pmlLayers - width)  / (float)pmlLayers), 1.0f),
			fminf(fmaxf(fmaxf(0.0f, 1.0f - (loc.y - 0.5f) / (float)pmlLayers), (loc.y - 0.5f + pmlLayers - height) / (float)pmlLayers), 1.0f)
		);

		float2 currentValues = make_float2(
			tex2D(velocityXTexRef, x + 0.5, y + 0.5),
			tex2D(velocityYTexRef, x + 0.5, y + 0.5)
		);

		float2 values = make_float2(
			currentValues.x + timeStep * (soundVelocity * grad.x / stepSize/* - pml.x * currentValues.x*/),
			currentValues.y + timeStep * (soundVelocity * grad.y / stepSize/* - pml.y * currentValues.y*/)
		);

		if(x < width - 1)
		{
			surf2Dwrite<float>(values.y, velocityYSurfRef, x * sizeof(float), y);
		}
		if(y < height - 1)
		{
			surf2Dwrite<float>(values.x, velocityXSurfRef, x * sizeof(float), y);
		}
		//float value = surf2Dread<float>(pressureSurfRef, x * sizeof(float), y);
	}
}

extern "C" __global__ void iteratePressure(float timeStep)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width - 1 && y < height - 1)
	{
		float2 loc = make_float2(x, y);
		float div = divergence(loc, velocityXTexRef, velocityYTexRef);

		//p = p + timeStep * (soundVelocity * (1 - g) * _d - pml * p + psi)
		float currentValue = tex2D(pressureTexRef, x + 0.5, y + 0.5);
		float values = currentValue + timeStep * (soundVelocity * div);
		surf2Dwrite<float>(values, pressureSurfRef, x * sizeof(float), y);

	}
}

extern "C" __global__ void iterateAux(float timeStep)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width - 1 && y < height - 1)
	{
		float2 loc = make_float2(x, y);

		float2 pml = make_float2(
			fminf(fmaxf(fmaxf(0.0f, 1.0f - loc.x / (float)pmlLayers), (loc.x + pmlLayers - width + 1) / (float)pmlLayers), 1.0f),
			fminf(fmaxf(fmaxf(0.0f, 1.0f - loc.y / (float)pmlLayers), (loc.y + pmlLayers - height + 1) / (float)pmlLayers), 1.0f)
		);

		float currentPsi = tex2D(psiTexRef, x + 0.5, y + 0.5);
		float currentPhi = tex2D(phiTexRef, x + 0.5, y + 0.5);
		float pressure = tex2D(pressureTexRef, x + 0.5, y + 0.5);

		float dvx = divergence_x(loc, velocityXTexRef);
		float dvy = divergence_x(loc, velocityYTexRef);

		currentPsi += timeStep * (soundVelocity * pml.x * dvy / stepSize - currentPsi * pml.y);
		currentPhi += timeStep * pml.y * (soundVelocity * dvx / stepSize + currentPsi - pressure * pml.x);

		surf2Dwrite<float>(currentPsi, psiSurfRef, x * sizeof(float), y);
		surf2Dwrite<float>(currentPhi, phiSurfRef, x * sizeof(float), y);

	}

	// psi = psi + timeStep * (soundVelocity * (1 - g) * px * (vy[:, 1 : ] - vy[:, : -1]) / stepSize - psi * py)
	// phi = phi + timeStep * py * (soundVelocity * (1 - g) * (vx[1:, : ] - vx[:-1, : ]) / stepSize + psi - p * px)
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
		surf2Dwrite<float>(value, forcesSurfRef, location.x * sizeof(float), location.y);
	}
}
