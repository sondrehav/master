#pragma once

#include "../JuceLibraryCode/JuceHeader.h"
#include "GLAbstractions.h"
#include "shader.h"
#include "PluginProcessor.h"

class TimelineDisplayGL : public Component, OpenGLRenderer
{
public:

	TimelineDisplayGL(ConvolutionReverbAudioProcessor& processor);
	~TimelineDisplayGL();

	TimelineDisplayGL(const TimelineDisplayGL&) = delete;
	TimelineDisplayGL& operator=(const TimelineDisplayGL&) = delete; // non copyable

	void setInputBuffer(AudioSampleBuffer& buffer, double sr);

	void shutdownOpenGL();
	void newOpenGLContextCreated() override;
	void renderOpenGL() override;
	void openGLContextClosing() override;

	void paint(Graphics& g) override;
	
	inline const size_t getBufferSizeUsed() { return bufferSizeUsed; }
private:

	const size_t datapointsPerSample = 32;
	const size_t maxSeconds = 20;
	const size_t bufferSize = maxSeconds * 44100 / datapointsPerSample; /* able to hold up to 20 seconds of data */

	size_t bufferSizeUsed = 0;
	float* dataL;
	float* dataR;

	std::unique_ptr<VertexBuffer> leftXBuffer = nullptr;
	std::unique_ptr<VertexBuffer> leftYBuffer = nullptr;
	std::unique_ptr<VertexArray> leftVertexArray = nullptr;

	std::unique_ptr<VertexBuffer> rightXBuffer = nullptr;
	std::unique_ptr<VertexBuffer> rightYBuffer = nullptr;
	std::unique_ptr<VertexArray> rightVertexArray = nullptr;

	std::unique_ptr<Shader> shader;

	OpenGLContext context;

	ConvolutionReverbAudioProcessor& processor;

};