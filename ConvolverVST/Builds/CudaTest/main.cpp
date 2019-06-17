#include <limits>
#include "../../Source/helper/ppm.h"
#define NOMINMAX
#include <windows.h>
#include "../../Source/Convolution.h"
#include <chrono>
#include <thread>
#include "../../Source/CUDAAbstractions.h"

int idivup(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

void console()
{
	FILE* fp;
	freopen_s(&fp, "conin$", "r", stdin);
	freopen_s(&fp, "conout$", "w", stdout);
	freopen_s(&fp, "conout$", "w", stderr);
	printf("Debugging Window:\n");
}

float* makeForces(int w, int h)
{
	float* data = new float[w*h];
	std::fill_n(data, w * h, 1.0f);

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++)
		{
			float fx = ((float)x / w - 0.5);
			float fy = ((float)y / h - 0.5);

			float vv = 0.1*0.1 - fx * fx - fy * fy;

			if (vv > 0)
			{
				vv = sqrt(vv);
			}
			else
			{
				vv = 0.0;
			}

			data[y * w + x] = 1000 * vv;

		}
	}
	return data;
}

int main()
{
	console();

	int w = 1001;
	int h = 1001;

	dim3 block = dim3(16, 16, 1);
	dim3 grid = dim3(idivup(w, block.x), idivup(h, block.y), 1);

	cuda::Module module("ptx/solver.ptx");

	float spatialStep = 0.1f;
	float speedOfSound = 343.0f;

	module.copyConstantToDevice("spatialStep", &spatialStep);
	module.copyConstantToDevice("speedOfSound", &speedOfSound);
	module.copyConstantToDevice("width", &w);
	module.copyConstantToDevice("height", &h);
	
	cuda::Texture2D::TextureProps props;
	props.numChannels = 2;

	cuda::Texture2D velocityDst(w, h, props);
	cuda::Texture2D pressureSrc(w, h);
	cuda::Texture2D forcesTex(w, h);
	cuda::Texture2D tempStorageTex(w, h);
	cuda::Texture2D aux(w, h, props);
	
	float* data = makeForces(w, h);
	forcesTex.setData(data);
	
	float dt = 1.0f /10000.0f;

	auto iteratePressure = *module.makeKernel<CUsurfObject, CUsurfObject, float>("iteratePressure");
	auto iterateAux = *module.makeKernel<CUsurfObject, CUsurfObject, CUsurfObject, float>("iterateAux");
	auto iterateVelocity = *module.makeKernel<CUsurfObject, CUsurfObject, float>("iterateVelocity");
	auto addForces = *module.makeKernel<CUsurfObject, CUsurfObject, CUsurfObject>("addForces");
	
	size_t numShared = (block.x + 2 * PADDING) * (block.y + 2 * PADDING);

	try {
		CUsurfObject pressure = pressureSrc.getSurfObject();
		CUsurfObject velocity = velocityDst.getSurfObject();
		CUsurfObject tempStorage = tempStorageTex.getSurfObject();
		CUsurfObject forces = forcesTex.getSurfObject();
		CUsurfObject auxSurf = aux.getSurfObject();

		CUDA_D(addForces(grid, block, 0, pressure, forces, tempStorage));
		CUDA_D(iterateVelocity(grid, block, numShared * sizeof(float), velocity, tempStorage, 0.5 * dt));
		CUDA_D(iterateAux(grid, block, numShared * sizeof(float), auxSurf, velocity, pressure, 0.5 * dt));

		const long numIters = 10000;
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < numIters; i++)
		{
			CUDA_D(iteratePressure(grid, block, numShared * sizeof(float2), pressure, velocity, dt));
			CUDA_D(addForces(grid, block, 0, pressure, forces, tempStorage));
			CUDA_D(iterateVelocity(grid, block, numShared * sizeof(float), velocity, tempStorage, dt));
			CUDA_D(iterateAux(grid, block, numShared * sizeof(float), auxSurf, velocity, pressure, dt));
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		float secondsPerIteration = (float)time / (numIters * 1e9);
		printf("time per iteration: %f seconds\nmax freq:           %f Hz\niterations:         %d\n", secondsPerIteration, 1.0 / secondsPerIteration, numIters);

	} catch (std::exception e)
	{
		printf("%s\n", e.what());
		exit(-1);
	}

	float* dat2a = pressureSrc.getData().get();
	float* dat3a = velocityDst.getData().get();

	std::string outputFile1 = std::string("output_pressure_") + std::to_string(ORDER) + std::string(".ppm");
	std::string outputFile2 = std::string("output_velocity_") + std::to_string(ORDER) + std::string(".ppm");

	write(outputFile1.c_str(), w, h, 1, dat2a, true);
	write(outputFile2.c_str(), w, h, 2, dat3a, true);
	write("input.ppm", w, h, 1, (float*) data, false);

	delete[] data;

}
