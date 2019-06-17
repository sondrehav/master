#include "SimulatorComponent.h"

SimulatorComponent::SimulatorComponent(ConvolutionReverbAudioProcessor& audioProcessor) : audioProcessor(audioProcessor)
{
	//audioProcessor.bufferChangedBroadcaster.addChangeListener(this);
	display = std::make_unique<SimulatorGLComponent>(audioProcessor);
	addAndMakeVisible(*display);
}

SimulatorComponent::~SimulatorComponent()
{
	display.reset();
}


void SimulatorComponent::resized()
{
	auto bounds = getLocalBounds();
	display->setBounds(bounds.reduced(60, 30));
}