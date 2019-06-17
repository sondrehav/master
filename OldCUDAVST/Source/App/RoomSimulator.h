#pragma once
#include "../JuceLibraryCode/JuceHeader.h"
#include <map>
#include <thread>

class RoomSimulator
{
public:
	RoomSimulator();
	~RoomSimulator()
	{
		delete[] ir;
	}

	std::thread* simulate(std::function<void(void)> cb);
	const AudioSampleBuffer* getBuffer(float x, float y, float theta);

	void setDomain(float* data, size_t xDim, size_t yDim, Rectangle<float> bounds)
	{
		this->dataBuffer = data;
		this->dataXDim = xDim;
		this->dataYDim = yDim;
		this->bounds = bounds;
	}

	void setStepSize(float x, float y) { stepSizeX = x; stepSizeY = y; }

	int getMaxFreq() const { return this->maxFreq; }
	void setMaxFreq(int maxFreq) { this->maxFreq = maxFreq; }

	float getLengthInSeconds() const { return lengthInSeconds; }
	void setLengthInSeconds(float lengthInSeconds) { this->lengthInSeconds = lengthInSeconds; }


private:
	Rectangle<float> bounds;
	float stepSizeX = 1, stepSizeY = 1;
	float* dataBuffer = nullptr;
	size_t dataXDim = 0, dataYDim = 0;

	float lengthInSeconds = 10;
	int maxFreq = 1000;

	std::map<int, AudioSampleBuffer*> buffers;

	float* ir;
	size_t irSize = 2;

};
