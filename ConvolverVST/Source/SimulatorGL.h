#pragma once

#include "../JuceLibraryCode/JuceHeader.h"
#include "PluginProcessor.h"
#include "GLAbstractions.h"
#include "shader.h"
#include "ChangeListenerHelper.h"

class SimulatorGLComponent : public Component, OpenGLRenderer
{
public:
	SimulatorGLComponent(ConvolutionReverbAudioProcessor& audioProcessor);
	~SimulatorGLComponent();

	SimulatorGLComponent(const SimulatorGLComponent&) = delete;
	SimulatorGLComponent& operator=(const SimulatorGLComponent&) = delete;

	void mouseDrag(const MouseEvent& event) override;
	void mouseDown(const MouseEvent& event) override;
	void mouseUp(const MouseEvent& event) override;
	void mouseWheelMove(const MouseEvent& event, const MouseWheelDetails& wheel) override;
	void mouseMove(const MouseEvent& event) override;

	void shutdownOpenGL();
	void newOpenGLContextCreated() override;
	void renderOpenGL() override;
	void openGLContextClosing() override;

	void paint(Graphics& g) override;

private:

	void renderPressureField(const float* matrix);
	void renderGeometry(const float* matrix);
	void renderPaintBrush(const float* matrix);

	void initializeTextures();
	void uninitializeTextures();

	OpenGLContext context;
	ConvolutionReverbAudioProcessor& processor;

	std::shared_ptr<Texture2D> texture = nullptr;
	std::shared_ptr<Texture1D> lutTexture = nullptr;
	
	std::shared_ptr<Texture2D> geometryTexture = nullptr;
	std::shared_ptr<Texture1D> geometryLutTexture = nullptr;
	std::unique_ptr<Shader> paintingShader = nullptr;

	std::unique_ptr<VertexBuffer> paintingBuffer = nullptr;
	std::unique_ptr<VertexArray> paintingArray = nullptr;

	std::unique_ptr<VertexBuffer> screenQuadBuffer = nullptr;
	std::unique_ptr<VertexArray> screenQuadArray = nullptr;
	std::unique_ptr<Shader> screenShader = nullptr;

	const float vertexData[5 * 6] = {
		 0.5f,  0.5f, 0.0f, 1.0f, 0.0f,  // top right
		 0.5f, -0.5f, 0.0f, 1.0f, 1.0f,  // bottom right
		-0.5f,  0.5f, 0.0f, 0.0f, 0.0f,  // top left 
		-0.5f, -0.5f, 0.0f, 0.0f, 1.0f,  // bottom left
		-0.5f,  0.5f, 0.0f, 0.0f, 0.0f,  // top left 
		 0.5f, -0.5f, 0.0f, 1.0f, 1.0f  // bottom right
	};

	const uint8_t lut[5 * 4] = {
		0x00,0x00,0x00,0xff,
		0x76,0x20,0x8c,0xff,
		0xf2,0x37,0x6c,0xff,
		0xe8,0x88,0x35,0xff,
		0xff,0xfa,0xd6,0xff
	};

	const uint8_t geometryLut[2 * 4] = {
		0x00,0x00,0x00,0x00,
		0x00,0x00,0x00,0x7f
	};

	ChangeListenerHelper* dimensionsChanged = nullptr;
	ChangeListenerHelper* displayChanged = nullptr;
	ChangeListenerHelper* geometryChanged = nullptr;

	int width, height;
	float paintSize = 90;
	float paintAmount = 1;
	float paintFalloff = 0.5;
	bool cursorDown = false;
	Point<float> paintBrushLocation;

};
