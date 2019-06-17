#pragma once

#include "Solver.h"

#include <functional>
#include <map>
#include <vector>

#define GLM_FORCE_PURE
#include <glm/glm.hpp>
#include <thread>
#include <mutex>
#include <algorithm>


#define SPEED_OF_SOUND 340

namespace compute{

struct PUBLIC_API Source
{
	Source(glm::vec2 position) : position(position) {}

	glm::vec2 position;
	float size = 2;

	bool operator<(const Source &ob) const
	{
		//(a.x < b.x) || (a.x == b.x && a.y < b.y);
		return (position.x < ob.position.x) || (position.x == ob.position.x && position.y < ob.position.y);
	}

};

struct PUBLIC_API Destination
{
	Destination(glm::vec2 position) : position(position) {}

	glm::vec2 position;

	bool operator<(const Destination &ob) const
	{
		return (position.x < ob.position.x) || (position.x == ob.position.x && position.y < ob.position.y);
	}
};


class PUBLIC_API Solver {

public:
	Solver(size_t sampleRate, glm::vec2 lowerBoundary, glm::vec2 upperBoundary, glm::vec2 stepSize = glm::vec2(0.015));
	~Solver();

	/**/
	bool addSource(const Source& source);

	/**/
	bool addDestination(const Destination& destination);

	/**/
	void reset();

	typedef std::map<std::pair<Source, Destination>, const float*> impulseContainer;
	/* Starts the simulation */
	void simulate(float simulationLengthInSeconds, const std::function<void(float)>& progress, const std::function<void(impulseContainer, size_t)>& complete, const std::function<void(impulseContainer, size_t)>& cancelled = nullptr);

	/* Stops the simulation */
	void cancel() { shouldQuit = true; }

	void getDimensions(size_t* width, size_t* height)
	{
		*width = (size_t) this->numSteps.x;
		*height = (size_t)this->numSteps.y;
	}

	void getContents(float* target);

	/**/
	void setImageData(float* data, size_t width, size_t height);

	/**/
	//void resize(size_t newWidth, size_t newHeight);

<<<<<<< Updated upstream
	float boundaryAmount = .05;

=======
>>>>>>> Stashed changes
private:

	/**/
	std::map<compute::Destination, float*> simulateSourceToDestinations(const Source& source, const std::function<void(float)>& progress);
	
	/**/
	float* resample(float* data, size_t oldWidth, size_t oldHeight, size_t newWidth, size_t newHeight);

	/**/
	std::vector<Source> sources;
	std::vector<Destination> destinations;

	glm::vec2 lowerBoundary;
	glm::vec2 upperBoundary;
	glm::vec2 stepSize;
	glm::tvec2<size_t> numSteps;
	const size_t sampleRate;
	size_t bufferLength = 0;

	std::thread* simulationThread = nullptr;
	bool shouldQuit = false;

	std::mutex lock;

	// CUDA stuff
	float* d_position;
	float* d_boundary;

	size_t pitch;
	const int blockSizeX = 16;
	const int blockSizeY = 16;

};

}