#pragma once

#include "../JuceLibraryCode/JuceHeader.h"
#include "PluginProcessor.h"
#include "SimulatorGL.h"

class SimulatorComponent : public Component
{
public:
	SimulatorComponent(ConvolutionReverbAudioProcessor& audioProcessor);
	~SimulatorComponent();

	void resized() override;

private:
	ConvolutionReverbAudioProcessor& audioProcessor;
	std::unique_ptr<SimulatorGLComponent> display;

};
