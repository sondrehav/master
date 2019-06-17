#include "Simulator.h"

#include <cuda_runtime.h>
#include "CUDAHelpers.h"
#include "CUDAKernels.h"
#include <algorithm>

int idivup(int a, int b)
{
	return (int)ceil((float)a / b);
}


compute::Solver::Solver(size_t sampleRate, glm::vec2 lowerBoundary, glm::vec2 upperBoundary, glm::vec2 stepSize) : sampleRate(sampleRate) {
	
	glm::vec2 volume = upperBoundary - lowerBoundary;
	this->numSteps.x = ceil(volume / stepSize).x;
	this->numSteps.y = ceil(volume / stepSize).y;
	this->stepSize = stepSize;
	this->lowerBoundary = lowerBoundary;
	this->upperBoundary = upperBoundary;

	// set up position
	size_t pitchBoundary;

	CUDA(cudaMallocPitch(&d_position, &pitch, this->numSteps.x * sizeof(float), this->numSteps.y));
	CUDA(cudaMallocPitch(&d_boundary, &pitchBoundary, this->numSteps.x * sizeof(float), this->numSteps.y));

	assert(pitchBoundary == pitch);

	CUDA(cudaMemset2D(d_position, pitch, 0x00000000, this->numSteps.x * sizeof(float), this->numSteps.y));
	CUDA(cudaMemset2D(d_boundary, pitch, 0x00000000, this->numSteps.x * sizeof(float), this->numSteps.y));

	// cudaMemset2D did not work :(
	float* ones = new float[this->numSteps.x * this->numSteps.y];
	std::fill(ones, ones + this->numSteps.x * this->numSteps.y, 1.0f);
	CUDA(cudaMemcpy2D(d_boundary, pitchBoundary, ones, this->numSteps.x * sizeof(float), this->numSteps.x * sizeof(float), this->numSteps.y, cudaMemcpyHostToDevice));
	CUDA(cudaDeviceSynchronize());
	delete [] ones;

}

compute::Solver::~Solver() {

	lock.lock();
	CUDA(cudaFree(d_position));
	CUDA(cudaFree(d_boundary));
	lock.unlock();
}


void compute::Solver::getContents(float* target)
{
<<<<<<< Updated upstream
	// todo: Fix mutual exclusion. Now flickering occurs when boundary has not been applied during simulation.
	//lock.lock();
	CUDA(cudaMemcpy2D(target, this->numSteps.x * sizeof(float), d_position, pitch, this->numSteps.x * sizeof(float), this->numSteps.y, cudaMemcpyDeviceToHost));
=======
	//lock.lock();
	CUDA(cudaMemcpy2D(target, this->numSteps.x * sizeof(float), d_boundary, pitch, this->numSteps.x * sizeof(float), this->numSteps.y, cudaMemcpyDeviceToHost));
>>>>>>> Stashed changes
	//lock.unlock();
}

std::map<compute::Destination, float*> compute::Solver::simulateSourceToDestinations(const Source& source, const std::function<void(float)>& progress)
{
	lock.lock();
	std::map<Destination, float*> result;

	/**/
	std::map<Destination, float*> deviceResults;
	
	// set up impulse response buffers
	for(Destination d : destinations)
	{
		result[d] = new float[this->bufferLength];
		float* d_buffer;
		CUDA(cudaMalloc(&d_buffer, sizeof(float) * bufferLength));
		deviceResults[d] = d_buffer;
	}

	// set up intermediate value buffers on device
	float* d_velocity;
	float* d_temp;
	float* d_tempVertical;
	float* d_forces;

	size_t pitchVelocity;
	size_t pitchTemp;
	size_t pitchTempVertical;
	size_t pitchForces;

	CUDA(cudaMallocPitch(&d_velocity, &pitchVelocity, this->numSteps.x * sizeof(float), this->numSteps.y));
	CUDA(cudaMallocPitch(&d_temp, &pitchTemp, this->numSteps.x * sizeof(float), this->numSteps.y));
	CUDA(cudaMallocPitch(&d_tempVertical, &pitchTempVertical, this->numSteps.x * sizeof(float), this->numSteps.y));
	CUDA(cudaMallocPitch(&d_forces, &pitchForces, this->numSteps.x * sizeof(float), this->numSteps.y));

	assert(pitch == pitchVelocity);
	assert(pitch == pitchTemp);
	assert(pitch == pitchTempVertical);
	assert(pitch == pitchForces);

	CUDA(cudaMemset2D(d_velocity, pitch, 0x00000000, this->numSteps.x * sizeof(float), this->numSteps.y));
	CUDA(cudaMemset2D(d_temp, pitch, 0x00000000, this->numSteps.x * sizeof(float), this->numSteps.y));
	CUDA(cudaMemset2D(d_tempVertical, pitch, 0x00000000, this->numSteps.x * sizeof(float), this->numSteps.y));
	CUDA(cudaMemset2D(d_forces, pitch, 0x00000000, this->numSteps.x * sizeof(float), this->numSteps.y));

	CUDA(cudaMemset2D(d_position, pitch, 0x00000000, this->numSteps.x * sizeof(float), this->numSteps.y));

	// set up CUDA compute dimensions
	dim3 gridSize = dim3(idivup(pitch / sizeof(float), blockSizeX), idivup(this->numSteps.y, blockSizeY), 1);
	dim3 blockSize = dim3(blockSizeX, blockSizeY, 1);

	// set up source and boundary
	float2 sourcePosition;
	sourcePosition.x = source.position.x - this->lowerBoundary.x;
	sourcePosition.y = source.position.y - this->lowerBoundary.y;

	float2 boundary;
	boundary.x = this->upperBoundary.x - this->lowerBoundary.x;
	boundary.y = this->upperBoundary.y - this->lowerBoundary.y;

	// Create a Gaussian 2d field for the initial impulse
//#define TEST_CONTINUAL_FORCE
#ifdef TEST_CONTINUAL_FORCE
	gaussianAdd << <gridSize, blockSize >> > (sourcePosition, boundary, source.size, 10e1, d_forces, this->numSteps.x, this->numSteps.y, pitch);
#else
	gaussianAdd << <gridSize, blockSize >> > (sourcePosition, boundary, source.size, 10e6, d_forces, this->numSteps.x, this->numSteps.y, pitch);
#endif
	CUDA(cudaDeviceSynchronize());
	CUDA(cudaPeekAtLastError());

	// set up initial velocities
	const float rhxsq = 1.0 / pow(this->stepSize.x, 2);
	const float rhysq = 1.0 / pow(this->stepSize.y, 2);
	const float timeStep = 1.0 / sampleRate;

	// v_{\frac{1}{2}} = v_0 + \frac{1}{2}k*(f + c^2\nabla^2u)
	convolveArrayVertical << <gridSize, blockSize >> > (d_position, d_tempVertical, pitch, this->numSteps.x, this->numSteps.y, 1.0 / 180.0);
	CUDA(cudaPeekAtLastError());
	convolveArrayHorizontal << <gridSize, blockSize >> > (d_position, d_temp, pitch, this->numSteps.x, this->numSteps.y, 1.0 / 180.0);
	CUDA(cudaPeekAtLastError());
	CUDA(cudaDeviceSynchronize());
	add << <gridSize, blockSize >> > (d_tempVertical, d_temp, d_temp, rhysq, rhxsq, pitch / sizeof(float), this->numSteps.y);
	CUDA(cudaPeekAtLastError());
	CUDA(cudaDeviceSynchronize());

	add << <gridSize, blockSize >> > (d_forces, d_temp, d_temp, 1, pow(SPEED_OF_SOUND, 2), pitch / sizeof(float), this->numSteps.y);
	CUDA(cudaPeekAtLastError());
	CUDA(cudaDeviceSynchronize());

	add << <gridSize, blockSize >> > (d_velocity, d_temp, d_velocity, 1, 0.5 * timeStep, pitch / sizeof(float), this->numSteps.y);
	CUDA(cudaPeekAtLastError());
	CUDA(cudaDeviceSynchronize());
	lock.unlock();

	// At this point the actual simulation can begin.
	
	for(int i = 0; i < this->bufferLength; i++)
	{
		// todo: fix the locks.
		lock.lock();

		add << <gridSize, blockSize >> > (d_velocity, d_position, d_position, timeStep, 1, pitch / sizeof(float), this->numSteps.y);
		CUDA(cudaPeekAtLastError());
		CUDA(cudaDeviceSynchronize());

<<<<<<< Updated upstream
		applyBoundary << <gridSize, blockSize >> > (d_position, d_boundary, d_position, boundaryAmount, pitch / sizeof(float), this->numSteps.y);
=======
		mult << <gridSize, blockSize >> > (d_position, d_boundary, d_position, 1, pitch / sizeof(float), this->numSteps.y);
>>>>>>> Stashed changes
		CUDA(cudaPeekAtLastError());
		CUDA(cudaDeviceSynchronize());

		// write positions to destinations
		for (Destination d : destinations)
		{
			float2 destPosition;
			destPosition.x = d.position.x - this->lowerBoundary.x;
			destPosition.y = d.position.y - this->lowerBoundary.y;

			float* d_res = deviceResults[d];
			writeBack<<<1, 1>>>(d_position, d_res, destPosition, boundary, i, pitch / sizeof(float), this->numSteps.y);
			CUDA(cudaPeekAtLastError());
			CUDA(cudaDeviceSynchronize());
		}

		// v_{n+\frac{3}{2}} = v_{n+\frac{1}{2}} + k*(f + c^2\nabla^2u)

		convolveArrayVertical << <gridSize, blockSize >> > (d_position, d_tempVertical, pitch, this->numSteps.x, this->numSteps.y, 1.0 / 180.0);
		CUDA(cudaPeekAtLastError());
		convolveArrayHorizontal << <gridSize, blockSize >> > (d_position, d_temp, pitch, this->numSteps.x, this->numSteps.y, 1.0 / 180.0);
		CUDA(cudaPeekAtLastError());
		CUDA(cudaDeviceSynchronize());
		add << <gridSize, blockSize >> > (d_tempVertical, d_temp, d_temp, rhysq, rhxsq, pitch / sizeof(float), this->numSteps.y);
		CUDA(cudaPeekAtLastError());
		CUDA(cudaDeviceSynchronize());

<<<<<<< Updated upstream
#ifdef TEST_CONTINUAL_FORCE
		// todo: remove this, only for testing...
		add << <gridSize, blockSize >> > (d_forces, d_temp, d_temp, 1, pow(SPEED_OF_SOUND, 2), pitch / sizeof(float), this->numSteps.y);
		CUDA(cudaPeekAtLastError());
		CUDA(cudaDeviceSynchronize());
#endif

=======
>>>>>>> Stashed changes
		add << <gridSize, blockSize >> > (d_velocity, d_temp, d_velocity, 1, timeStep * pow(SPEED_OF_SOUND, 2), pitch / sizeof(float), this->numSteps.y);
		CUDA(cudaPeekAtLastError());
		CUDA(cudaDeviceSynchronize());

		lock.unlock();

		/*
		add << <gridSize, blockSize >> > (d_velocity, d_temp, d_velocity, 1, timeStep, pitchPosition / sizeof(float), this->numSteps.y);
		CUDA(cudaPeekAtLastError());
		CUDA(cudaDeviceSynchronize());
		*/

		// Report progress to calling object.
		if(i % (this->bufferLength / 10) == 0)
		{
			progress((float)i / this->bufferLength);
		}

		// Exit if requested.
		if (this->shouldQuit) break;

	}

	lock.lock();

	// copy back...
	for (Destination d : destinations)
	{
		CUDA(cudaMemcpy(result[d], deviceResults[d], sizeof(float) * this->bufferLength, cudaMemcpyDeviceToHost));
	}

	// ... and clean up
	for (Destination d : destinations)
	{
		CUDA(cudaFree(deviceResults[d]));
	}

	CUDA(cudaFree(d_temp));
	CUDA(cudaFree(d_tempVertical));
	CUDA(cudaFree(d_forces));
	CUDA(cudaFree(d_velocity));

	lock.unlock();

	// And we are done! :)

	return result;
}

<<<<<<< Updated upstream
=======
/*
void compute::Solver::resize(size_t newWidth, size_t newHeight)
{
	if (simulationThread != nullptr) {
		simulationThread->join();
		delete simulationThread;
		simulationThread = nullptr;
	}
	lock.lock();
	
	lock.unlock();
}
*/

>>>>>>> Stashed changes
void compute::Solver::setImageData(float* data, size_t width, size_t height)
{
	if (simulationThread != nullptr) {
		cancel();
		simulationThread->join();
		delete simulationThread;
		simulationThread = nullptr;
	}
	lock.lock();

<<<<<<< Updated upstream
	// We need the data to fit the domain so we resize to current grid size.
	data = resample(data, width, height, numSteps.x, numSteps.y);

	CUDA(cudaMemcpy2D(d_boundary, pitch, data, numSteps.x * sizeof(float), numSteps.x * sizeof(float), numSteps.y, cudaMemcpyHostToDevice));
=======
	data = resample(data, width, height, numSteps.x, numSteps.y);

	/*float min = std::numeric_limits<float>::max();
	float max = std::numeric_limits<float>::min();
	for(int y = 0; y < newNumSteps.y; y++)
	{
		for (int x = 0; x < newNumSteps.x; x++)
		{
			min = std::min(data[y * newNumSteps.x + x], min);
			max = std::max(data[y * newNumSteps.x + x], max);
		}
	}*/

	CUDA(cudaMemset2D(d_position, pitch, 0x00000000, this->numSteps.x * sizeof(float), this->numSteps.y));

	size_t pitchBoundary;
	CUDA(cudaFree(d_boundary));
	CUDA(cudaMallocPitch(&d_boundary, &pitchBoundary, this->numSteps.x * sizeof(float), this->numSteps.y));
	assert(pitchBoundary == pitch);
	CUDA(cudaMemcpy2D(d_boundary, pitchBoundary, data, numSteps.x * sizeof(float), numSteps.x, numSteps.y, cudaMemcpyHostToDevice));
>>>>>>> Stashed changes
	lock.unlock();
}
