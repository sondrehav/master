#pragma once

#include "../JuceLibraryCode/JuceHeader.h"
#include "../Simulator/Convolve.h"
#include <thread>

class RoomProcessor
{

public:

	RoomProcessor(int width, int height, float size, int sampleRate) : solver(width, height, size, sampleRate / 2)
	{
		float* forces = new float[width * height];
		std::fill_n(forces, width * height, 0);
		for (int y = 10; y < 20; y++) for (int x = 10; x < 20; x++) forces[y * width + x] = 100000;
		solver.setForces(forces);
		delete[] forces;
	}

	~RoomProcessor()
	{
		
	}

	void setSourceL(const Point<float>& p) { sourceL = p; }
	void setSourceR(const Point<float>& p) { sourceR = p; }
	void setDestL(const Point<float>& p) { destL = p; }
	void setDestR(const Point<float>& p) { destR = p; }

	void prepareToPlay(double sampleRate, int samplesPerBlock);
	void releaseResources();
	void processBlock(AudioBuffer<float>&);

	const float* getContents(int* width, int* height);

private:
	compute::Solver solver;
	std::thread* thread;

	void calculate();

	Point<float> sourceL;
	Point<float> sourceR;
	Point<float> destL;
	Point<float> destR;

	bool shouldClose = false;

};
