#pragma once

#include "program/program.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda/cudaTexture.h"
#include "gl/textureRenderer.h"
#include <glm/glm.hpp>
#include "gl/editorRenderer.h"
#include "../audio/AudioFile.h"


class WaveSolver : public Program
{
public:


	WaveSolver() : Program(700, 700)
	{
	}

	~WaveSolver(){}

	void init() override;
	void loop(float dt) override;
	void destroy() override;

	void onKeyUp(int keyNum, int mods) override;
	void onKeyDown(int keyNum, int mods) override;
	void onFrameTimeExceededLimit(float) override { printf("Frame!\n"); }

	void onMouseMove(double x, double y) override;
	void onMouseDown(double x, double y, int mouseNum, int mods) override;
	void onMouseUp(double x, double y, int mouseNum, int mods) override;
	void onMouseScroll(double dx, double dy) override;
	void onMouseExit() override;

	union vec2i
	{
		glm::ivec2 hostVec;
		int2 cudaVec;
	};

protected:
	void onResized(int width, int height) override;
private:
	
	bool initCUDA();
	void initCUDAConstants();

	float* zeros(int numChannels, int width, int height);
	void copyConstantToDevice(const std::string& name, const void* value);
	
	void beginUI();
	void renderUI(int width, int height, double dt);
	
	void resetSimulation();
	void simulationLoop();
	void initializeSimulation();
	void clearGeometry();

	CUmodule module;
	CUdevice device;
	CUcontext context;

	uint64_t iteration = 0;
	
	CUdeviceptr waveOutput = 0;
	size_t numOutputSamples = 0;

	CUdeviceptr waveInput = 0;
	size_t numInputSamples = 0;

	CUfunction iteratePressureFunction;
	CUfunction iterateVelocityFunction;
	CUfunction firstIterationVelocityFunction;
	CUfunction drawLineFunction;
	CUfunction sampleAtFunction;
	CUfunction sampleInFunction;

	CUtexref pressureTexRef;
	CUsurfref pressureSurfRef;

	CUtexref velocityTexRef;
	CUsurfref velocitySurfRef;

	CUtexref geometryTexRef;
	CUsurfref geometrySurfRef;

	CUtexref forcesTexRef;
	CUsurfref forcesSurfRef;

	std::unique_ptr<RWCUDATexture2D> pressureTexture;
	std::unique_ptr<RWCUDATexture2D> velocityTexture;

	std::unique_ptr<RWCUDATexture2D> geometryTexture;
	std::unique_ptr<RWCUDATexture2D> forcesTexture;

	std::unique_ptr<TextureRenderer> renderer;
	std::unique_ptr<EditorRenderer> editorRenderer;

	dim3 dimBlock = dim3(8, 8, 1);
	dim3 dimGrid;

	float currentFrequency = 0.0f;

	const int width = 420, height = 420;
	const int pmlLayers = 40;
	const int textureWidth = width + 2 * pmlLayers, textureHeight = height + 2 * pmlLayers;

	int stepsPerSample = 8;
	const int sampleRate = 44100;
	float soundVelocity = 340.0f;
	float stepSize = 7.70975e-3;
	float timeStep = 1.0f/(sampleRate*stepsPerSample);
	float pmlMax = 0.1;
	float wallAbsorbtion = 0.1;

	float tail = 2.0f; // 2 seconds.
	float amp = 9.0f; // 2 seconds.

	bool simulate = false;

	glm::mat4 uvMatrix;

	/* Painting */

	BrushState brushState = BrushState::NoneBrushState;
	PaintOperation paintOperation = PaintOperation::NonePaintOperation;

	glm::vec2 initialLineLocation;
	glm::vec2 mousePosition;
	float brushSize = 10.0f;

	/**/

	vec2i sourcePositionsL;
	vec2i sourcePositionsR;
	vec2i destinationPositionsL;
	vec2i destinationPositionsR;

	AudioFile<float>* inputFile = nullptr;

	const int numInputChannels = 2;
	const int numOutputChannels = 2;

};


