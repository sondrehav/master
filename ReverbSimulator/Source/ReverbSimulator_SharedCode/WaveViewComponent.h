

#pragma once

#include "../../JuceLibraryCode/JuceHeader.h"
#include "PluginProcessor.h"

class WaveViewComponent  : public OpenGLAppComponent, public Timer
{
public:
	WaveViewComponent(ReverbSimulatorAudioProcessor&);
    ~WaveViewComponent();

	void initialise() override;
	void render() override;
	void shutdown() override;

    void paint (Graphics&) override;
    void resized() override;

	void timerCallback() override;

private:
    ReverbSimulatorAudioProcessor& processor;

	std::unique_ptr<OpenGLShaderProgram> shader;

	const float vertexData[5*6] = { 
		 0.5f,  0.5f, 0.0f, 1.0f, 0.0f,  // top right
		 0.5f, -0.5f, 0.0f, 1.0f, 1.0f,  // bottom right
		-0.5f,  0.5f, 0.0f, 0.0f, 0.0f,  // top left 
		-0.5f, -0.5f, 0.0f, 0.0f, 1.0f,  // bottom left
		-0.5f,  0.5f, 0.0f, 0.0f, 0.0f,  // top left 
		 0.5f, -0.5f, 0.0f, 1.0f, 1.0f  // bottom right
	};

	/*
	const Colour lut[5] = {
		Colour(0xff000000),
		Colour(0xff00163a),
		Colour(0xff870018),
		Colour(0xffffd400),
		Colour(0xfffff7d3)
	};*/

	const Colour lut[5] = {
		Colour(0xffffffff),
		Colour(0xff00ff00),
		Colour(0xff000000),
		Colour(0xffff0000),
		Colour(0xffffffff)
	};
	
	GLuint vertexBufferId;
	GLuint textureBufferId;
	GLuint lutBufferId;
	bool updateImage = false;

	compute::Solver* solver;
	compute::CLContext* context;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (WaveViewComponent)
};
