#pragma once

#include "../JuceLibraryCode/JuceHeader.h"
#include "PluginProcessor.h"
#include <glm/gtc/matrix_transform.hpp>

class RoomComponent : public OpenGLAppComponent {
public:
	RoomComponent(ConvolutionReverbAudioProcessor&);
	~RoomComponent();

	void initialise() override;
	void render() override;
	void shutdown() override;

	void paint(Graphics&) override;
	void resized() override;

	void mouseWheelMove(const MouseEvent& event, const MouseWheelDetails& wheel) override;

	void mouseDrag(const MouseEvent& event) override;

	void mouseUp(const MouseEvent& event) override
	{
		Logger::writeToLog("mouseUp");
		bool t = isMouseButtonDown();
		if(mouseState == dragging && !ModifierKeys::getCurrentModifiers().isLeftButtonDown())
		{
			mouseState = none;
		}
	}
	

	enum MouseState
	{
		none = 0,
		dragging = 1,
		rotating = 2
	};

	struct Camera
	{

		Camera(int width, int height) : width(width), height(height)
		{

		}
		Camera() : Camera(600,400) {}

		glm::mat4 projection()
		{
			return glm::mat4(
				glm::vec4(zoom / width, 0.0, 0.0, 2 * zoom * position.x / width),
				glm::vec4(0.0, zoom / height, 0.0, 2 * zoom * position.y / height),
				glm::vec4(0.0, 0.0, 1.0, 0.0),
				glm::vec4(0.0, 0.0, 0.0, 1.0)
			);
		}

		glm::vec2 position = glm::vec2(0.0f, 0.0f);
		float zoom = 1.0f;
		int width;
		int height;
		

	};

private:

	void loadGridTexture();
	

	ConvolutionReverbAudioProcessor& processor;

	std::unique_ptr<OpenGLShaderProgram> shader;
	std::unique_ptr<OpenGLShaderProgram> gridShader;

	int projectionMatrixLocation = -1;
	int modelMatrixLocation = -1;

	const float vertexData[5 * 6] = {
		 0.5f,  0.5f, 0.0f, 1.0f, 0.0f,  // top right
		 0.5f, -0.5f, 0.0f, 1.0f, 1.0f,  // bottom right
		-0.5f,  0.5f, 0.0f, 0.0f, 0.0f,  // top left 
		-0.5f, -0.5f, 0.0f, 0.0f, 1.0f,  // bottom left
		-0.5f,  0.5f, 0.0f, 0.0f, 0.0f,  // top left 
		 0.5f, -0.5f, 0.0f, 1.0f, 1.0f  // bottom right
	};

	const Colour lut[5] = {
		Colour(0xff00ff00),
		Colour(0xffe5f257),
		Colour(0xff000000),
		Colour(0xfff78c27),
		Colour(0xffff0000)
	};

	ReadWriteLock dataLock;
	std::function<void(size_t, size_t)> listener;

	GLuint vertexBufferId;
	GLuint textureBufferId;
	GLuint lutTextureId;
	GLuint gridTextureId;

	size_t width, height;
	float* data;

	Camera camera;
	MouseState mouseState = none;
	glm::vec2 mouseEventLast;

	JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(RoomComponent)
};
