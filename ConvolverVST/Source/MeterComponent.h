#pragma once

#include "../JuceLibraryCode/JuceHeader.h"

class MeterComponent : public Component, public Timer
{
public:
	MeterComponent(NormalisableRange<float> range, std::function<Array<float>()> valueGetter, std::function<String(float)> labels = nullptr);
	~MeterComponent();

	void paint(Graphics& g) override;
	void resized() override;
	void visibilityChanged() override;

	void setBackgroundColour(Colour colour) { mBackgroundColour = colour; }
	void setGradientColour(ColourGradient colour) { mColourGradient = colour; }
	void setInverted(bool inverted) { mInverted = inverted; }
	void setLabels(std::function<String(float)> l) { mLabels = l; }

private:
	ColourGradient mColourGradient;
	Colour mBackgroundColour = Colour(0xff424549);
	Colour mOutlineColour = Colour(0x80000000);
	Colour mMarkerColour = Colour(0x20ffffff);

	std::function<String(float)> mLabels;
	NormalisableRange<float> mRange;

	std::function<Array<float>()> mQueryForValue;
	bool mInverted = false;
	int mLabelSpacing = 5;

	void timerCallback() override;

};