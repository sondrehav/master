/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include "../JuceLibraryCode/JuceHeader.h"
#include "PluginProcessor.h"
#include "RoomComponent.h"
#include "AudioWaveformComponent.h"

//==============================================================================
/**
*/
class ConvolutionReverbAudioProcessorEditor  : public AudioProcessorEditor
{
public:
    ConvolutionReverbAudioProcessorEditor (ConvolutionReverbAudioProcessor&);
    ~ConvolutionReverbAudioProcessorEditor();

    //==============================================================================
    void paint (Graphics&) override;
    void resized() override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    ConvolutionReverbAudioProcessor& processor;

	class DummyComponent : public Component
	{
	public:
		DummyComponent() {}
		~DummyComponent() {}

		void paint(Graphics& g) override
		{
			g.fillAll(Colours::grey);
			g.setColour(Colours::red);
			g.drawRect(getLocalBounds());
		}

	};

	class FlexWrapperComponent : public Component
	{
	public:
		FlexWrapperComponent()
		{
			flex.flexDirection = FlexBox::Direction::row;
			flex.justifyContent = FlexBox::JustifyContent::spaceBetween;
			flex.alignItems = FlexBox::AlignItems::stretch;
			flex.flexWrap = FlexBox::Wrap::noWrap;
		}

		FlexWrapperComponent(FlexBox::Direction direction)
		{
			flex.flexDirection = direction;
			flex.justifyContent = FlexBox::JustifyContent::spaceBetween;
			flex.alignItems = FlexBox::AlignItems::stretch;
			flex.flexWrap = FlexBox::Wrap::noWrap;
		}

		~FlexWrapperComponent() {}

		FlexBox flex;
		void resized() override {
			flex.performLayout(getLocalBounds());
		}

		void paint(Graphics& g) override
		{
			g.setColour(Colours::red);
			g.drawRect(getLocalBounds());
		}

		void addItem(Component& component)
		{
			this->addAndMakeVisible(component);
			flex.items.add(component);
		}

		void addItem(FlexItem flexItem)
		{
			this->addAndMakeVisible(flexItem.associatedComponent);
			flex.items.add(flexItem);
		}

	};

	class RotationSlider : public Slider
	{
	public:
		RotationSlider()
		{
			setSliderStyle(RotaryVerticalDrag);
			setTextBoxStyle(TextBoxBelow, false, 90, 20);
		}
	private:
	};

	class ChangeListenerWrapper : public ChangeListener
	{
	public:
		ChangeListenerWrapper(std::function<void(ChangeBroadcaster*)> l) : fn(l) {}
		void changeListenerCallback(ChangeBroadcaster* source) { fn(source); }
	private:
		const std::function<void(ChangeBroadcaster*)> fn;
	};

	ChangeListenerWrapper* audioWaveformListener;

	Slider testSlider;

	RoomComponent domain;
	AudioWaveformComponent waveForm;

	TextButton openFileButton;
	TextButton saveFileButton;
	TextButton simulateButton;

	RotationSlider xPositionSlider;
	RotationSlider yPositionSlider;
	RotationSlider thetaSlider;
	RotationSlider drySlider;
	RotationSlider wetSlider;
	
	RotationSlider stretchSlider;
	RotationSlider pitchSlider;
	RotationSlider lowPassSlider;
	RotationSlider highPassSlider;
	RotationSlider separationSlider;

	FlexWrapperComponent mainFlex;
	FlexWrapperComponent knobsLeftFlex;
	FlexWrapperComponent knobsRightFlex;
	FlexWrapperComponent midFlex;
	FlexWrapperComponent bottomControlsFlex;
	FlexWrapperComponent saveLoadFlex;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ConvolutionReverbAudioProcessorEditor)
};
