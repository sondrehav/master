/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include "../JuceLibraryCode/JuceHeader.h"
#include "PluginProcessor.h"
#include "TimelineDisplayComponent.h"
#include "MeterComponent.h"
#include "SimulatorComponent.h"
#include "TimedLabel.h"

//==============================================================================
/**
*/
class ConvolutionReverbAudioProcessorEditor  : public AudioProcessorEditor, private ChangeListener
{
public:
    ConvolutionReverbAudioProcessorEditor (ConvolutionReverbAudioProcessor&);
    ~ConvolutionReverbAudioProcessorEditor();

    //==============================================================================
    void resized() override;

    void paint(Graphics& g) override;

    void changeListenerCallback(ChangeBroadcaster* source) override;

	

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    ConvolutionReverbAudioProcessor& processor;

	TextButton openFile;
	Slider drySlider, wetSlider;
	ToggleButton toggleMode;

	TimedLabel refreshRateCounter;
	TimedLabel iterationCounter;

	ComboBox dimensionChooser;
	
	TextButton startSimulation;
	TextButton stopSimulation;
	TextButton pauseSimulation;

	TimelineDisplayComponent timelineComponent;
	SimulatorComponent simulatorComponent;

	MeterComponent inputMeter;
	MeterComponent outputMeter;

	std::unique_ptr<AudioProcessorValueTreeState::SliderAttachment> wetSliderAttachment;
	std::unique_ptr<AudioProcessorValueTreeState::SliderAttachment> drySliderAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ConvolutionReverbAudioProcessorEditor)
};
