#include <cstdio>
#include <cuda.h>
#include <driver_types.h>
#include "../helper_math.h"
#include "../Convolution.h"

__constant__ int width;
__constant__ int height;
__constant__ float speedOfSound;
__constant__ float spatialStep;
__constant__ int numPMLLayers = 10;
__constant__ float pmlMax = 10;

/**
 * Gets the index for the shared memory given padding, position and dimension
 */
template<int P>
__device__ __inline__ int sharedMemIndex(int2 local, int blockDimX)
{
	return (blockDimX + 2 * P) * (local.y + P) + local.x + P;
}

/**
 * Recursive template base
 */
template<typename T, typename First>
__device__ __inline__ T readSummed(int2 global, First surface)
{
	return surf2Dread<T>(surface, global.x * sizeof(T), global.y, cudaBoundaryModeClamp);
}

/**
 * Recursive template. Reads every surface object and adds them together.
 */
template<typename T, typename First, typename ... Rest>
__device__ __inline__ T readSummed(int2 global, First first, Rest ... rest)
{
	return readSummed<T, First>(global, first) + readSummed<T, Rest...>(global, rest...);
}

/**
 * Write stuff to shared memory. Padded values as well.
 */
template<typename T, int P, typename ... Surfaces>
__device__ __inline__ void writeSummedToSharedMem(int2 global, int2 local, int2 blockDim, Surfaces... surfaces)
{
	extern __shared__ __align__(sizeof(T)) unsigned char sharedMemory[];
	T *sharedMem = reinterpret_cast<T *>(sharedMemory);

	sharedMem[sharedMemIndex<P>(local, blockDim.x)] = readSummed<T, Surfaces...>(global, surfaces...);

	if (local.x < P)                    sharedMem[sharedMemIndex<P>(make_int2(local.x - P, local.y), blockDim.x)] = readSummed<T, Surfaces...>(make_int2(global.x - P, global.y), surfaces...);
	else if (local.x >= blockDim.x - P) sharedMem[sharedMemIndex<P>(make_int2(local.x + P, local.y), blockDim.x)] = readSummed<T, Surfaces...>(make_int2(global.x + P, global.y), surfaces...);

	if (local.y < P)                    sharedMem[sharedMemIndex<P>(make_int2(local.x, local.y - P), blockDim.x)] = readSummed<T, Surfaces...>(make_int2(global.x, global.y - P), surfaces...);
	else if (local.y >= blockDim.y - P) sharedMem[sharedMemIndex<P>(make_int2(local.x, local.y + P), blockDim.x)] = readSummed<T, Surfaces...>(make_int2(global.x, global.y + P), surfaces...);
	
}

template<typename T, int P>
__device__ __inline__ T readFromSharedMem(int2 local, int2 blockDim)
{
	extern __shared__ __align__(sizeof(T)) unsigned char sharedMemory[];
	T *sharedMem = reinterpret_cast<T *>(sharedMemory);
	return sharedMem[sharedMemIndex<P>(local, blockDim.x)];
}

template<int P>
__device__ __inline__ float2 gradient(int2 local, int blockDim)
{
	extern __shared__ __align__(sizeof(float)) unsigned char sharedMemory[];
	float *sharedMem = reinterpret_cast<float *>(sharedMemory);

	const float conv[CONVOLUTION_SIZE] = CONVOLUTION;
	float sumX = 0;
	float sumY = 0;

	#pragma unroll
	for (int i = -P; i <= P; i++)
	{
		float conv_i = conv[i + P];
		sumX += sharedMem[sharedMemIndex<P>(local + make_int2(i, 0), blockDim)] * conv_i;
		sumY += sharedMem[sharedMemIndex<P>(local + make_int2(0, i), blockDim)] * conv_i;
	}
	return make_float2(sumX, sumY);
}

template<int P>
__device__ __inline__ float derivativeX(int2 local, int blockDim)
{
	extern __shared__ __align__(sizeof(float2)) unsigned char sharedMemory[];
	float2 *sharedMem = reinterpret_cast<float2 *>(sharedMemory);

	const float conv[CONVOLUTION_SIZE] = CONVOLUTION;
	float sum = 0.0f;

#pragma unroll
	for (int i = -P; i <= P; i++)
	{
		float conv_i = conv[i + P];
		float dx = sharedMem[sharedMemIndex<P>(local + make_int2(i, 0), blockDim)].x * conv_i;
		sum += dx;
	}
	return sum;
}

template<int P>
__device__ __inline__ float derivativeY(int2 local, int blockDim)
{
	extern __shared__ __align__(sizeof(float2)) unsigned char sharedMemory[];
	float2 *sharedMem = reinterpret_cast<float2 *>(sharedMemory);

	const float conv[CONVOLUTION_SIZE] = CONVOLUTION;
	float sum = 0.0f;

#pragma unroll
	for (int i = -P; i <= P; i++)
	{
		float conv_i = conv[i + P];
		float dy = sharedMem[sharedMemIndex<P>(local + make_int2(0, i), blockDim)].y * conv_i;
		sum += dy;
	}
	return sum;
}

template<int P>
__device__ __inline__ float divergence(int2 local, int blockDim)
{
	return derivativeX<P>(local, blockDim) + derivativeY<P>(local, blockDim);
}

__device__ __inline__ float2 pml(int2 global)
{

	float px = fmaxf(0.0f, fmaxf(1.0f - global.x / numPMLLayers, (global.x - width + numPMLLayers) / numPMLLayers));
	float py = fmaxf(0.0f, fmaxf(1.0f - global.y / numPMLLayers, (global.y - height + numPMLLayers) / numPMLLayers));
	return (1.0 - make_float2(px, py)) * pmlMax;
}

extern "C" __global__ void iterateVelocity(CUsurfObject velocitySurface, CUsurfObject pressureSurfaceSrc, CUsurfObject geometrySurfaceSrc, float dt)
{

	int2 dim = make_int2(blockDim.x, blockDim.y);
	int2 local = make_int2(threadIdx.x, threadIdx.y);
	int2 global = make_int2(blockIdx.x * dim.x + local.x, blockIdx.y * dim.y + local.y);
	float2 pmlValue = pml(global);

	writeSummedToSharedMem<float, PADDING, CUsurfObject>(global, local, dim, pressureSurfaceSrc);
	__syncthreads();

	float2 grad = gradient<PADDING>(local, dim.x) / spatialStep;
	float2 currentVelocity = surf2Dread<float2>(velocitySurface, global.x * sizeof(float2), global.y, cudaBoundaryModeClamp);

	float geom = surf2Dread<float>(geometrySurfaceSrc, global.x * sizeof(float), global.y, cudaBoundaryModeClamp);
	float2 newVelocity = ((1 - geom) * speedOfSound * grad - currentVelocity * pmlValue) * dt + currentVelocity;

	if (global.x < width && global.y < height)
	{
		surf2Dwrite<float2>(newVelocity, velocitySurface, (int)sizeof(float2)*global.x, global.y, cudaBoundaryModeClamp);
	}

}

extern "C" __global__ void initialVelocity(CUsurfObject velocitySurface, CUsurfObject forcesSurfaceSrc, CUsurfObject geometrySurfaceSrc, float dt)
{

	int2 dim = make_int2(blockDim.x, blockDim.y);
	int2 local = make_int2(threadIdx.x, threadIdx.y);
	int2 global = make_int2(blockIdx.x * dim.x + local.x, blockIdx.y * dim.y + local.y);
	float2 pmlValue = pml(global);

	writeSummedToSharedMem<float, PADDING, CUsurfObject>(global, local, dim, forcesSurfaceSrc);
	__syncthreads();

	float2 grad = gradient<PADDING>(local, dim.x) / spatialStep;
	float2 currentVelocity = surf2Dread<float2>(velocitySurface, global.x * sizeof(float2), global.y, cudaBoundaryModeClamp);

	float geom = surf2Dread<float>(geometrySurfaceSrc, global.x * sizeof(float), global.y, cudaBoundaryModeClamp);
	float2 newVelocity = ((1 - geom) * speedOfSound * grad - currentVelocity * pmlValue) * dt + currentVelocity;


	if (global.x < width && global.y < height)
	{
		surf2Dwrite<float2>(newVelocity, velocitySurface, (int)sizeof(float2)*global.x, global.y, cudaBoundaryModeClamp);
	}

}

extern "C" __global__ void iterateAux(CUsurfObject auxSurface, CUsurfObject velocitySurface, CUsurfObject pressureSurfaceSrc, float dt)
{
	int2 dim = make_int2(blockDim.x, blockDim.y);
	int2 local = make_int2(threadIdx.x, threadIdx.y);
	int2 global = make_int2(blockIdx.x * dim.x + local.x, blockIdx.y * dim.y + local.y);
	float2 pmlValue = pml(global);

	writeSummedToSharedMem<float2, PADDING, CUsurfObject>(global, local, dim, velocitySurface);
	__syncthreads();

	float dx = derivativeX<PADDING>(local, dim.x);
	float dy = derivativeY<PADDING>(local, dim.x);

	float2 currentAux = surf2Dread<float2>(auxSurface, global.x * sizeof(float2), global.y, cudaBoundaryModeClamp);
	float currentPressure = surf2Dread<float>(pressureSurfaceSrc, global.x * sizeof(float), global.y, cudaBoundaryModeClamp);

	float newAuxX = (speedOfSound * pmlValue.x * dy - currentAux.x * pmlValue.y) * dt + currentAux.x;
	float newAuxY = pmlValue.y * (speedOfSound * dx + currentAux.x - currentPressure * pmlValue.x) * dt + currentAux.y;

	if (global.x < width && global.y < height)
	{
		surf2Dwrite<float2>(make_float2(newAuxX, newAuxY), auxSurface, (int)sizeof(float2)*global.x, global.y, cudaBoundaryModeClamp);
	}

}

extern "C" __global__ void iteratePressure(CUsurfObject pressureSurface, CUsurfObject velocitySurfaceSrc, CUsurfObject auxSurfaceSrc, float dt)
{

	int2 dim = make_int2(blockDim.x, blockDim.y);
	int2 local = make_int2(threadIdx.x, threadIdx.y);
	int2 global = make_int2(blockIdx.x * dim.x + local.x, blockIdx.y * dim.y + local.y);
	float2 pmlValues = pml(global);
	float pmlValue = fmaxf(pmlValues.x + pmlValues.y, pmlMax);

	writeSummedToSharedMem<float2, PADDING, CUsurfObject>(global, local, dim, velocitySurfaceSrc);

	__syncthreads();

	float div = divergence<PADDING>(local, dim.x) / spatialStep;
	float current = surf2Dread<float>(pressureSurface, global.x * sizeof(float), global.y, cudaBoundaryModeClamp);
	float2 currentAux = surf2Dread<float2>(auxSurfaceSrc, global.x * sizeof(float2), global.y, cudaBoundaryModeClamp);
	
	float newValue = (speedOfSound * div + currentAux.x + currentAux.y - pmlValue * current) * dt + current;

	if (global.x < width && global.y < height)
	{
		surf2Dwrite<float>(newValue, pressureSurface, (int)sizeof(float)*global.x, global.y, cudaBoundaryModeClamp);
	}

}
