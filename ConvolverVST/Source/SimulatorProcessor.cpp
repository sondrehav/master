#include "SimulatorProcessor.h"
#include "Convolution.h"
#define NOMINMAX
#include <windows.h>
#include "helper/ppm.h"

int idivup(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

#ifdef _DEBUG
void console()
{
	AllocConsole();
	FILE* fp;
	freopen_s(&fp, "conin$", "r", stdin);
	freopen_s(&fp, "conout$", "w", stdout);
	freopen_s(&fp, "conout$", "w", stderr);
	printf("Debugging Window:\n");
}
#endif


SimulatorProcessor::SimulatorProcessor(unsigned int width, unsigned int height, float sr) 
	: width(width), height(height), sampleRate(sr)
{
#ifdef _DEBUG
	console();
#endif
	module = std::make_unique<cuda::Module>("ptx/solver.ptx");

	module->makeCurrent();

	grid = dim3(idivup(width, block.x), idivup(height, block.y), 1);

	iteratePressureKernel = module->makeKernel<CUsurfObject, CUsurfObject, CUsurfObject, float>("iteratePressure");
	iterateVelocityKernel = module->makeKernel<CUsurfObject, CUsurfObject, CUsurfObject, float>("iterateVelocity");
	initialVelocityKernel = module->makeKernel<CUsurfObject, CUsurfObject, CUsurfObject, float>("initialVelocity");
	iterateAuxKernel = module->makeKernel<CUsurfObject, CUsurfObject, CUsurfObject, float>("iterateAux");

	geometryField = std::make_unique<float[]>(width * height);

	sharedMemSize = (block.x + 2 * PADDING) * (block.y + 2 * PADDING);
	timeStep = 1.0 / sampleRate;
}

SimulatorProcessor::~SimulatorProcessor()
{
	if (simulationState != Stopped) stop();
	iteratePressureKernel.reset();
	iterateVelocityKernel.reset();
	initialVelocityKernel.reset();
	iterateAuxKernel.reset();
	module.reset();
	geometryField.reset();
}

float* makeForces(int w, int h)
{
	float* data = new float[w*h];
	
	std::fill_n(data, w * h, 0.0f);
	data[w*h / 2 + w / 2] = 1e3;

	printMinmax(data, w, h);
	return data;
}

void SimulatorProcessor::start()
{
	
	if (simulationState == Started) throw std::exception("Simulator already started.");

	if (simulationState == Stopped)
	{
		module->makeCurrent();

		cuda::Texture2D::TextureProps props;
		props.numChannels = 2;

		pressure = std::make_shared<cuda::Texture2D>(width, height);
		velocity = std::make_shared<cuda::Texture2D>(width, height, props);
		forces = std::make_shared<cuda::Texture2D>(width, height);
		aux = std::make_shared<cuda::Texture2D>(width, height, props);
		geometry = std::make_shared<cuda::Texture2D>(width, height);
		geometry->setData(geometryField.get());

		auto f = makeForces(width, height);
		forces->setData(f);
		delete[] f;

		pressureSurfObject = pressure->getSurfObject();
		velocitySurfObject = velocity->getSurfObject();
		auxSurfObject = aux->getSurfObject();
		forcesSurfObject = forces->getSurfObject();
		geometrySurfObject = geometry->getSurfObject();

		pressureField.reset();
		pressureField = std::make_unique<float[]>(width * height);
		outputFieldChanged.sendChangeMessage();

		spatialStep = speedOfSound / sampleRate;
		//spatialStep = 7.70975e-3;
		
		module->copyConstantToDevice("spatialStep", &spatialStep);
		module->copyConstantToDevice("speedOfSound", &speedOfSound);
		module->copyConstantToDevice("width", &width);
		module->copyConstantToDevice("height", &height);
		module->copyConstantToDevice("pmlMax", &pmlMax);
		module->copyConstantToDevice("numPMLLayers", &numPMLLayers);

		this->timeStep = 1.0 / (sampleRate * 2);
		//this->timeStep = 1.0 / 10000.0;
		
		CUDA_D((*initialVelocityKernel)(grid, block, sharedMemSize * sizeof(float), velocitySurfObject, forcesSurfObject, geometrySurfObject, 0.5 * timeStep));

		iteration = 0;

	}

	simulationState = Started;
	simulatorStateChanged.sendChangeMessage();
	simulationThread = new std::thread(&SimulatorProcessor::simulate, this);

}

void SimulatorProcessor::stop()
{
	if (simulationState == Stopped) throw std::exception("Simulation already stopped.");

	if(simulationState == Started)
	{
		simulationState = Stopped;
		simulationThread->join();
		delete simulationThread;
		simulationThread = nullptr;
	}

	simulationState = Stopped;
	simulatorStateChanged.sendChangeMessage();
	
	module->makeCurrent();
	pressure = nullptr;
	velocity = nullptr;
	forces = nullptr;
	aux = nullptr;

	iteration = 0;
}

void SimulatorProcessor::pause()
{
	if (simulationState != Started) throw std::exception("Simulation not running.");
	simulationState = Paused;
	simulatorStateChanged.sendChangeMessage();
	simulationThread->join();
	delete simulationThread;
	simulationThread = nullptr;
}


void SimulatorProcessor::simulate()
{
	long internalIteration = 0;
	
	module->makeCurrent();
	int readBackNS = 1.0e9  / readBackRate;

	auto start = std::chrono::high_resolution_clock::now();
	while(simulationState == Started)
	{
		CUDA_D((*iteratePressureKernel)(grid, block, sharedMemSize * sizeof(float2), pressureSurfObject, velocitySurfObject, auxSurfObject, timeStep));
		CUDA_D((*iterateAuxKernel)(grid, block, sharedMemSize * sizeof(float2), auxSurfObject, velocitySurfObject, pressureSurfObject, timeStep));
		CUDA_D((*iterateVelocityKernel)(grid, block, sharedMemSize * sizeof(float), velocitySurfObject, pressureSurfObject, geometrySurfObject, timeStep));

		internalIteration++;
		iteration++;

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		if(elapsed > readBackNS)
		{
			start = end;
			secondsPerIteration = (float)elapsed / (internalIteration * 1e9);
#ifdef _DEBUG
			printf("time per iteration: %f seconds\nmax freq:           %f Hz\niteration:          %d Hz\n", secondsPerIteration, 1.0 / secondsPerIteration, internalIteration);
#endif
			internalIteration = 0;
			readBack();
		}

	}
}


/* Getters and setters */
void SimulatorProcessor::setWidth(unsigned width)
{
	if (simulationState != Stopped) throw std::exception("Simulation needs to be stopped in order to set variables.");
	this->width = width;
	geometryField.reset();
	geometryField = std::make_unique<float[]>(width * height);
	dimensionsChanged.sendChangeMessage();
}

void SimulatorProcessor::setHeight(unsigned height)
{
	if (simulationState != Stopped) throw std::exception("Simulation needs to be stopped in order to set variables.");
	this->height = height;
	geometryField.reset();
	geometryField = std::make_unique<float[]>(width * height);
	dimensionsChanged.sendChangeMessage();
}

void SimulatorProcessor::setDimensions(unsigned width, unsigned height)
{
	if (simulationState != Stopped) throw std::exception("Simulation needs to be stopped in order to set variables.");
	this->width = width;
	this->height = height;
	geometryField.reset();
	geometryField = std::make_unique<float[]>(width * height);
	dimensionsChanged.sendChangeMessage();
}

void SimulatorProcessor::setSampleRate(float sampleRate)
{
	if (simulationState != Stopped) throw std::exception("Simulation needs to be stopped in order to set variables.");
	this->sampleRate = sampleRate;
}

void SimulatorProcessor::setSpatialStep(float spatialStep)
{
	if (simulationState != Stopped) throw std::exception("Simulation needs to be stopped in order to set variables.");
	this->spatialStep = spatialStep;
}


void SimulatorProcessor::readBack()
{
	const std::function<void(std::shared_ptr<float[]>)> f = [&](std::shared_ptr<float[]> data)
	{
		pressureField = data;
		outputFieldChanged.sendChangeMessage();
	};
	pressure->getDataAsync(f);
}


void SimulatorProcessor::drawGeometryAt(Point<float> pos, float size, float amount, float falloff)
{
	for (int i = std::max<float>(pos.y - size, numPMLLayers); i < std::min<float>(pos.y + size, height - numPMLLayers); i++)
	for (int j = std::max<float>(pos.x - size, numPMLLayers); j < std::min<float>(pos.x + size, width - numPMLLayers); j++)
	{
		float xx = (j - pos.x + size) / (2 * size);
		float yy = (i - pos.y + size) / (2 * size);
		float rx = pow(2.0 * xx - 1.0, 2.0);
		float ry = pow(2.0 * yy - 1.0, 2.0);
		float value = std::min((1.0 - rx - ry) / falloff, 1.0) * abs(amount);
		if(value > 0)
		{
			geometryField[i*width+j] = std::clamp<float>(geometryField[i*width + j] + value * (amount > 0 ? 1 : -1), 0, 1);
		}
	}
	geometryChanged.sendChangeMessage();
}
