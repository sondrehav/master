#include "Simulator.h"
#include <deque>
#include <algorithm>

bool compute::Solver::addSource(const Source& source)
{
	sources.push_back(source);
	return true;
}

bool compute::Solver::addDestination(const Destination& destination)
{
	destinations.push_back(destination);
	return true;
}

void compute::Solver::reset()
{
	shouldQuit = true;
}

void compute::Solver::simulate(float simulationLengthInSeconds, const std::function<void(float)>& progress, const std::function<void(impulseContainer, size_t)>& complete, const std::function<void(impulseContainer, size_t)>& cancelled)
{
	if(simulationThread != nullptr)
	{
		cancel();
		simulationThread->join();
		delete simulationThread;
		simulationThread = nullptr;
	}
	shouldQuit = false;
	simulationThread = new std::thread([this, simulationLengthInSeconds, cancelled, progress, complete]()
	{
		impulseContainer container;
		this->bufferLength = simulationLengthInSeconds * sampleRate;
		int i = 0;
		for(Source s : sources)
		{
			auto result = simulateSourceToDestinations(s, [&](float p)
			{
				float chunkProgress = (float)i / sources.size();
				float totalProgress = chunkProgress + p / sources.size();
				progress(totalProgress);
			});
			for(auto item : result)
			{
				container[std::pair<Source, Destination>(s, item.first)] = item.second;
			}
			i++;
			if (shouldQuit)
			{
				cancelled(container, this->bufferLength);
				return;
			}
		}
		complete(container, this->bufferLength);
		
	});
}

float* compute::Solver::resample(float* data, size_t oldWidth, size_t oldHeight, size_t newWidth, size_t newHeight)
{
	
	if (oldWidth == newWidth && oldHeight == newHeight) return data;
	
	float* newData = new float[newWidth * newHeight];

	for(size_t y = 0; y < newHeight; y++)
	{
		float y1 = (oldHeight - 1) * (float)y / (newHeight - 1);
		size_t indexY1 = floor(y1);
		size_t indexY2 = ceil(y1);
		float lerpY = indexY2 - y1;
		assert(lerpY >= 0 && lerpY <= 1.0);

		for (size_t x = 0; x < newWidth; x++)
		{
			float x1 = (oldWidth- 1) * (float)x / (newWidth - 1);
			size_t indexX1 = floor(x1);
			size_t indexX2 = ceil(x1);
			float lerpX = indexX2 - x1;
			assert(lerpX >= 0 && lerpX <= 1.0);

			float f0 = data[indexY1 * oldWidth + indexX1] * lerpX + data[indexY1 * oldWidth + indexX2] * (1.0 - lerpX);
			float f1 = data[indexY2 * oldWidth + indexX1] * lerpX + data[indexY2 * oldWidth + indexX2] * (1.0 - lerpX);
			float f = f0 * lerpY + f1 * (1.0 - lerpY);

			newData[newWidth * y + x] = f;
		}
	}

#if 0
	int i, j;
	FILE *fp;
	fopen_s(&fp, "first.ppm", "wb"); /* b - binary mode */
	(void)fprintf(fp, "P6\n%d %d\n255\n", (int)newWidth, (int)newHeight);
	for (j = 0; j < newHeight; ++j)
	{
		for (i = 0; i < newWidth; ++i)
		{
			static unsigned char color[3];
			color[0] = (int)(255 * newData[newWidth*j + i]) % 256;  /* red */
			color[1] = (int)(255 * newData[newWidth*j + i]) % 256;  /* green */
			color[2] = (int)(255 * newData[newWidth*j + i]) % 256;  /* blue */
			(void)fwrite(color, 1, 3, fp);
		}
	}
	(void)fclose(fp);
#endif

	delete[] data;
	return newData;

}