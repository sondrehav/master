#pragma once

#include "../JuceLibraryCode/JuceHeader.h"

class AudioWaveformComponent : public Component, public ChangeListener
{
	
public:
	AudioWaveformComponent(size_t imgSamplesPerSample, ConvolutionReverbAudioProcessor& p) : imgSamplesPerSample(imgSamplesPerSample), processor(p)
	{
		processor.addAudioBufferChangedListener(this);
	}

	~AudioWaveformComponent()
	{
		processor.removeAudioBufferChangedListener(this);
	}

	void setData(const float* data, size_t length)
	{


		size_t width = ceil(length / imgSamplesPerSample);
		const size_t height = 64;
		img = new Image(Image::PixelFormat::ARGB, width, height, true);

		Graphics g(*img);

		const float* readPointer = data;

		float* resampled = new float[width];

		float min = std::numeric_limits<float>::max();
		float max = std::numeric_limits<float>::min();

		for(int sample = 0; sample < length; sample++)
		{
			min = std::min(readPointer[sample], min);
			max = std::max(readPointer[sample], max);
		}

		Logger::writeToLog("min: " + String(min) + "; max: " + String(max));

		float scaling = 1.0 / std::max(-min, max);

		for(int sample = 0; sample < width; sample++)
		{
			float in = readPointer[(int)(length * (float)sample / width)] * scaling;
			resampled[sample] = in;
		}

		g.setColour(Colours::red);
		float lastY = height * resampled[0] / 2 + height / 2;
		for(int x = 1; x < width; x++)
		{
			//float y = (int)(std::clamp<float>(1.0 - (resampled[x] + 48) / (48 + 6), 0, 1) * img->getHeight());

			float y = height * std::clamp<float>(resampled[x], -1, 1) / 2 + height / 2;

			Path p;
			p.startNewSubPath(x - 1, lastY);
			p.lineTo(x, y);
			p.lineTo(x, height / 2);
			p.lineTo(x - 1, height / 2);
			p.closeSubPath();
			g.fillPath(p);
			lastY = y;
		}
		delete[] resampled;

		postCommandMessage(10);

	}

	void handleCommandMessage(int commandId) override
	{
		if(commandId == 10)
		{
			repaint();
		}
	}

	void paint(Graphics& g) override
	{
		if(img != nullptr)
		{
			Rectangle<int> bounds = getLocalBounds();
			AffineTransform t;
			t = t.scaled((float)bounds.getWidth() / img->getWidth(), (float)bounds.getHeight() / img->getHeight());
			g.drawImageTransformed(*img, t);
		}
		g.drawFittedText("Hello, World!", 0, 0, getWidth(), getHeight(), Justification::centred, 0);
	}

	void changeListenerCallback(ChangeBroadcaster* source) override
	{
		AudioSampleBuffer* buffer = processor.getIR(0, 0);
		const float* data = buffer->getWritePointer(0);
		setData(data, buffer->getNumSamples());
	}

private:
	Image* img = nullptr;
	const size_t imgSamplesPerSample;
	ConvolutionReverbAudioProcessor& processor;

};
