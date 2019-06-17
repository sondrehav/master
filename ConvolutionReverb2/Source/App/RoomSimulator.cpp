#include "RoomSimulator.h"
#include <cmath>
#include <thread>

int main()
{

	float* data = new float[5 * 5]
	{
		1,1,1,1,1,
		1,0,0,0,1,
		1,0,0,0,1,
		1,0,0,0,1,
		1,1,1,1,1
	};
	RoomSimulator s;
	s.setDomain(data, 5, 5, Rectangle<float>(1, 1));
	s.setStepSize(.1, .1);
	s.setLengthInSeconds(5);
	s.setMaxFreq(1000);
	std::thread* thread = s.simulate([&]()
	{
		printf("Simulation done!\n");
	});


	thread->join();
	delete thread;
	printf("Exiting!\n");

}

RoomSimulator::RoomSimulator()
{
	ir = new float[irSize] {1, 0};
}


std::thread* RoomSimulator::simulate(std::function<void(void)> cb)
{
	int numXBuffers = ceil(bounds.getWidth() / stepSizeX);
	int numYBuffers = ceil(bounds.getHeight() / stepSizeY);
	int numSamples = maxFreq * lengthInSeconds;

	float* kernel = new float[7] { 0.01111111, -0.15, 1.5, -2.72222222, 1.5, -0.15, 0.01111111 };

	std::function<void(void)> fn = [&]()
	{

		cb();
	};
	std::thread* t = new std::thread(fn);
	return t;
}

const AudioSampleBuffer* RoomSimulator::getBuffer(float x, float y, float theta)
{
	return nullptr;
}