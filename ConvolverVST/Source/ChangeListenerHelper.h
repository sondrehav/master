#pragma once

#include "../JuceLibraryCode/JuceHeader.h"

class ChangeListenerHelper : public ChangeListener
{
public:
	explicit ChangeListenerHelper(const std::function<void(ChangeBroadcaster*)>& fn)
		: fn(fn)
	{
	}

	void changeListenerCallback(ChangeBroadcaster* source) override
	{
		fn(source);
	}

private:
	std::function<void(ChangeBroadcaster*)> fn;

};