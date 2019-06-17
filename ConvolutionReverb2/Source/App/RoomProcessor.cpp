#include "RoomProcessor.h"
#include <chrono>

void RoomProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
	/*if (thread != nullptr)
	{
		shouldClose = true;
		thread->join(); 
		delete thread; 
		shouldClose = false;
	}
	thread = new std::thread(this->calculate);*/
}

void RoomProcessor::processBlock(AudioBuffer<float>& b)
{
	const float* ptr = b.getReadPointer(0);
	size_t length = b.getNumSamples();
	solver.processBlock(ptr, length);
}

void RoomProcessor::releaseResources()
{
	shouldClose = true;
	if (thread != nullptr) { thread->join(); delete thread; }
}

void RoomProcessor::calculate()
{
	typedef std::chrono::high_resolution_clock Clock;
	double fps = 0.0;
	while(!shouldClose)
	{

		// 1. wait for process block

		auto start = Clock::now();
		solver.step();
		auto end = Clock::now();
		
		std::chrono::duration<double, std::milli> elapsed = end - start;
		double desired = solver.getTimestep() * 1000;
		double sleepTime = std::max<double>(desired - elapsed.count(), 0.0);
		std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(sleepTime));

		double total = elapsed.count() + sleepTime;
		fps = 0.1 * (1000.0 / total) + 0.9 * fps;
		
	}
}

const float* RoomProcessor::getContents(int* width, int* height)
{
	*width = solver.getWidth();
	*height = solver.getHeight();
	return solver.getContents();
}
