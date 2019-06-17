#pragma once

#include "../JuceLibraryCode/JuceHeader.h"

class TimedLabel : public Label, Timer
{
public:
	TimedLabel(float refreshRate) : refreshRate(refreshRate)
	{
	}

	void timerCallback() override
	{
		String v = getValueFn();
		setText(v, dontSendNotification);
	}

	void visibilityChanged() override
	{
		if (!this->isVisible() && isTimerRunning()) stopTimer();
		else startTimerHz((int)refreshRate);
	}

	std::function<String()> getValueFn;
	float refreshRate;

};