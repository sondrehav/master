#include "main.h"
#include "cudaDebug.h"
#include "helper_math.h"


void WaveSolver::resetSimulation()
{
	simulate = false;

	float* data = zeros(1, textureWidth, textureHeight);
	pressureTexture->setData(data, GL_RED, GL_FLOAT);
	velocityTexture->setData(data, GL_RED, GL_FLOAT);
	delete[] data;

	data = zeros(1, width, height);
	forcesTexture->setData(data, GL_RED, GL_FLOAT);
	delete[] data;

	iteration = 0;
	timeStep = 1.0f / (sampleRate*stepsPerSample);

	forceSwapBuffers();

	float halfTimeStep = timeStep;
	void *args[1] = { (void*)&halfTimeStep };

	CUDA_D(cuLaunchKernel(firstIterationVelocityFunction, dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, 0, NULL, args, NULL));
	CUDA(cudaDeviceSynchronize());
}

void WaveSolver::simulationLoop()
{
	float amp_ = pow(10, amp);
	void *pressureKernelArgs[1] = { (void*)&timeStep };

	auto start = std::chrono::high_resolution_clock::now();
	int num = 0;
	while(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() < 1000.0 / getMaxFPS() && simulate) {
		int sample = iteration / stepsPerSample;
		if (sample < numInputSamples && iteration % stepsPerSample == 0)
		{
			void *sampleInArgs[3] = { (void*)&waveInput, (void*)&sample, (void*)&amp_ };
			CUDA_D(cuLaunchKernel(sampleInFunction, 1, 1, 1, numInputChannels, 1, 1, 0, NULL, sampleInArgs, NULL));
		}

		CUDA_D(cuLaunchKernel(iteratePressureFunction, dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, 0, NULL, pressureKernelArgs, NULL));

		void *velocityKernelArgs[4] = { (void*)&timeStep, (void*)&waveInput, (void*)&sample, (void*)&wallAbsorbtion };
		CUDA_D(cuLaunchKernel(iterateVelocityFunction, dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, 0, NULL, velocityKernelArgs, NULL));

		if (sample < numOutputSamples && iteration % stepsPerSample == 0)
		{
			void *sampleOutArgs[2] = { (void*)&waveOutput, (void*)&sample };
			CUDA_D(cuLaunchKernel(sampleAtFunction, 1, 1, 1, numOutputChannels, 1, 1, 0, NULL, sampleOutArgs, NULL));
		}
		iteration++;
		num++;
		if (sample >= numOutputSamples) simulate = false;
	}
	currentFrequency = 0.9 * currentFrequency + 0.1 * (num / stepsPerSample) / frameTime;
}

void WaveSolver::initializeSimulation()
{
	assert(simulate == false);
	assert(inputFile != nullptr);

	if (inputFile->getSampleRate() != sampleRate)
	{
		printf("Warning: input file does not have %d as sample rate. It has %d. Will result in pitched sound.\n", sampleRate, inputFile->getSampleRate());
	}

	numInputSamples = inputFile->getNumSamplesPerChannel();
	numOutputSamples = inputFile->getNumSamplesPerChannel() + tail * sampleRate;

	if (waveOutput != 0) CUDA_D(cuMemFree(waveOutput));
	if (waveInput != 0) CUDA_D(cuMemFree(waveInput));

	CUDA_D(cuMemAlloc_v2(&waveOutput, numOutputChannels * numOutputSamples * sizeof(float)));
	CUDA_D(cuMemAlloc_v2(&waveInput, numInputChannels * numInputSamples * sizeof(float)));

	float* inputDataConcatinated = new float[numInputSamples * numInputChannels];
	std::memset(inputDataConcatinated, 0.0f, numInputSamples * numInputChannels * sizeof(float));

	for(int i = 0; i < numInputChannels; i++)
	{
		std::memcpy(inputDataConcatinated + numInputSamples * i, inputFile->samples[std::min(inputFile->getNumChannels() - 1, i)].data(), numInputSamples * sizeof(float));
	}

	CUDA_D(cuMemcpyHtoD_v2(waveInput, inputDataConcatinated, numInputSamples * numInputChannels * sizeof(float)));

	delete[] inputDataConcatinated;

	copyConstantToDevice("numInputSamples", &numInputSamples);
	copyConstantToDevice("numOutputSamples", &numOutputSamples);

	sourcePositionsL.hostVec.x = width / 8;
	sourcePositionsL.hostVec.y = height / 8;
	sourcePositionsR.hostVec.x = width / 8;
	sourcePositionsR.hostVec.y = 7 * height / 8;

	destinationPositionsL.hostVec.x = 7 * width / 8;
	destinationPositionsL.hostVec.y = height / 8;
	destinationPositionsR.hostVec.x = 7 * width / 8;
	destinationPositionsR.hostVec.y = 7 * height / 8;

	vec2i inputLocations[2] = { sourcePositionsL, sourcePositionsR };
	vec2i outputLocations[2] = { destinationPositionsL, destinationPositionsR };

	copyConstantToDevice("inputLocations", inputLocations);
	copyConstantToDevice("outputLocations", outputLocations);

	resetSimulation();

}

void WaveSolver::clearGeometry()
{
	float* data = zeros(1, width, height);
	geometryTexture->setData(data, GL_RED, GL_FLOAT);
	delete[] data;
}
