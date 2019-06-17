#pragma once

#include "CUDAAbstractions.h"

#include "../JuceLibraryCode/JuceHeader.h"
#include <thread>

class SimulatorProcessor
{
public:
	SimulatorProcessor(unsigned int width, unsigned int height, float sampleRate);

	SimulatorProcessor(const SimulatorProcessor&) = delete;
	SimulatorProcessor& operator=(const SimulatorProcessor&) = delete;
	~SimulatorProcessor();

	void start();
	void stop();
	void pause();

	void setWidth(unsigned width);
	void setHeight(unsigned height);
	void setDimensions(unsigned width, unsigned height);
	void setSampleRate(float sampleRate);
	void setSpatialStep(float spatialStep);

	unsigned getWidth() const { return width; }
	unsigned getHeight() const { return height; }
	float getSampleRate() const	{ return sampleRate; }
	float getSpatialStep() const { return spatialStep; }
	float getTimeStep() const { return timeStep; }

	float getSecondsPerIteration() const { return secondsPerIteration; }
	float getNumIterations() const { return iteration; }
	void drawGeometryAt(Point<float> pos, float size, float amount, float falloff);

	enum SimulationState
	{
		Stopped, Started, Paused
	};

	SimulationState getSimulationState() const { return simulationState; }

	ChangeBroadcaster& getOutputFieldChanged() { return outputFieldChanged; }
	ChangeBroadcaster& getDimensionsChanged() { return dimensionsChanged; }
	ChangeBroadcaster& getSimulatorStateChanged() { return simulatorStateChanged; }
	ChangeBroadcaster& getGeometryChanged() { return geometryChanged; }

	std::shared_ptr<float[]> getPressureFieldData() { return pressureField; }
	std::shared_ptr<float[]> getGeometryData() { return geometryField; }

private:
	void simulate();
	void readBack();

	std::unique_ptr<cuda::Module> module = nullptr;
	std::shared_ptr<cuda::Kernel<CUsurfObject, CUsurfObject, CUsurfObject, float>> iteratePressureKernel = nullptr;
	std::shared_ptr<cuda::Kernel<CUsurfObject, CUsurfObject, CUsurfObject, float>> iterateVelocityKernel = nullptr;
	std::shared_ptr<cuda::Kernel<CUsurfObject, CUsurfObject, CUsurfObject, float>> initialVelocityKernel = nullptr;
	std::shared_ptr<cuda::Kernel<CUsurfObject, CUsurfObject, CUsurfObject, float>> iterateAuxKernel = nullptr;
	
	std::shared_ptr<cuda::Texture2D> pressure = nullptr;
	std::shared_ptr<cuda::Texture2D> velocity = nullptr;
	std::shared_ptr<cuda::Texture2D> forces = nullptr;
	std::shared_ptr<cuda::Texture2D> aux = nullptr;
	std::shared_ptr<cuda::Texture2D> geometry = nullptr;

	CUsurfObject pressureSurfObject = 0;
	CUsurfObject velocitySurfObject = 0;
	CUsurfObject auxSurfObject = 0;
	CUsurfObject forcesSurfObject = 0;
	CUsurfObject geometrySurfObject = 0;

	unsigned int width, height;
	float sampleRate, spatialStep = 0.1;
	float timeStep;
	const float speedOfSound = 343.0f;
	float pmlMax = 10.0f;
	int numPMLLayers = 20;

	size_t sharedMemSize;
	unsigned long iteration = 0;

	dim3 grid;
	dim3 block = dim3(8, 8, 1);

	std::thread* simulationThread = nullptr;

	SimulationState simulationState = Stopped;

	ChangeBroadcaster outputFieldChanged;
	ChangeBroadcaster dimensionsChanged;
	ChangeBroadcaster simulatorStateChanged;
	ChangeBroadcaster geometryChanged;

	double readBackRate = 10.0; // Hz..
	float secondsPerIteration = 0;

	std::shared_ptr<float[]> pressureField;
	std::shared_ptr<float[]> geometryField;

};	
