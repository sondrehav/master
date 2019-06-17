#pragma once

#include "../JuceLibraryCode/JuceHeader.h"
#include "PluginProcessor.h"
#include "shader.h"
#include "GLAbstractions.h"
#include <mutex>
#include "TimelineDisplayGL.h"

class TimelineDisplayComponent : public Component, public ChangeListener
{
public:
	TimelineDisplayComponent(ConvolutionReverbAudioProcessor& processor);
	TimelineDisplayComponent(const TimelineDisplayComponent&) = delete;
	TimelineDisplayComponent& operator=(const TimelineDisplayComponent&) = delete; // non copyable

	~TimelineDisplayComponent();

	void changeListenerCallback(ChangeBroadcaster* source) override;
	void resized() override;
	void paint(Graphics& g) override;
	void visibilityChanged() override;

private:
	ConvolutionReverbAudioProcessor& audioProcessor;
	std::unique_ptr<TimelineDisplayGL> display;

};
