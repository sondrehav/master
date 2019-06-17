#include "MeterComponent.h"

MeterComponent::MeterComponent(NormalisableRange<float> range, std::function<Array<float>()> valueGetter, std::function<String(float)> labels):
	mQueryForValue(valueGetter), mRange(range), mLabels(labels)
{
	mColourGradient.addColour(0, Colour(0xff00285e));
	mColourGradient.addColour(0.5, Colour(0xff23ba2a));
	mColourGradient.addColour(0.75, Colour(0xfffce323));
	mColourGradient.addColour(1, Colour(0xffff0000));
	mColourGradient.isRadial = false;
	mColourGradient.point1 = Point<float>(0, 0);
	mColourGradient.point2 = Point<float>(0, 1);
}

MeterComponent::~MeterComponent()
{
	stopTimer();
}


void MeterComponent::paint(Graphics& g)
{

	Rectangle<int> bounds = getLocalBounds();
	g.setColour(mOutlineColour);
	g.drawRect(bounds, 2.0f);

	bounds.reduce(2, 2);
	g.setColour(mBackgroundColour);
	g.fillRect(bounds);

	{
		Array<float> values = mQueryForValue();
		Rectangle<int> meterBounds = Rectangle<int>(bounds);
		mColourGradient.point1 = Point<float>(0, bounds.getHeight());
		g.setGradientFill(mColourGradient);
		int width = meterBounds.getWidth() / values.size();
		for(float value : values)
		{
			value = 1.0 - mRange.convertTo0to1(mRange.snapToLegalValue(value));
			int height = (int)(value*bounds.getHeight());
			if (!mInverted) g.fillRect(meterBounds.removeFromLeft(width).withTop(height).reduced(1, 0));
			else g.fillRect(meterBounds.removeFromLeft(width).withBottom(height).reduced(1, 0));
		}
	}

	{
		float labelSize = std::min<float>(bounds.getWidth(), bounds.getHeight());
		g.setColour(mMarkerColour);
		Font f = g.getCurrentFont();
		f.setSizeAndStyle(10.0f, Font::FontStyleFlags::bold, 1.0f, 0);
		g.setFont(f);

		for (float i = mRange.start - fmod(mRange.start, mLabelSpacing); i <= mRange.end - fmod(mRange.end, mLabelSpacing); i += mLabelSpacing)
		{
			float value = 1.0 - mRange.convertTo0to1(i);
			float height = (int)(value*bounds.getHeight());
			g.fillRect(4, (int)height, bounds.getWidth() - 4, i == 0 ? 2 : 1);
			g.drawText(mLabels(i), 0, height-labelSize, labelSize, labelSize, Justification::bottomRight, false);
		}
	}
	
	
}

void MeterComponent::resized(){}

void MeterComponent::timerCallback()
{
	repaint();
}

void MeterComponent::visibilityChanged()
{
	if(isVisible()) startTimerHz(30);
	else stopTimer();
}

