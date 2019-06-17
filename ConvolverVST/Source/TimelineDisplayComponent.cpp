#include <glad/glad.h>
#include "TimelineDisplayComponent.h"
#include "debug.h"
#include <string>

const std::string vertexShaderSource = "#version 450 core\n\nlayout (location = 0) in float x;\nlayout (location = 1) in float y;\n\nuniform float scaling_x = 1.0;\n\nvoid main(){\n	float yy = 2 * (y + 48) / (6 + 48) - 1;\n	gl_Position = vec4(2 * x * scaling_x - 1, yy, 0.0, 1.0);\n}\n";
const std::string geometryShaderSource = "#version 450 core\n\nlayout (lines) in;\nlayout (triangle_strip, max_vertices = 4) out;\n\nvoid main(){\n	\n	vec4 v1 = gl_in[0].gl_Position;\n	vec4 v2 = gl_in[1].gl_Position;\n	vec4 vb1 = vec4(v1.x, -1.0f, v1.z, v1.w);\n	vec4 vb2 = vec4(v2.x, -1.0f, v2.z, v2.w);\n\n	gl_Position = v1;\n	EmitVertex();\n	gl_Position = v2;\n	EmitVertex();\n	gl_Position = vb1;\n	EmitVertex();\n	gl_Position = vb2;\n	EmitVertex();\n	EndPrimitive();\n\n}\n";
const std::string fragmentShaderSource = "#version 450 core\n\nlayout (location = 0) out vec4 color;\n\nuniform vec4 waveformColor = vec4(1.0f);\n\nvoid main() {\n	color = waveformColor;\n}";

TimelineDisplayComponent::TimelineDisplayComponent(ConvolutionReverbAudioProcessor& processor) :
	audioProcessor(processor)
{
	audioProcessor.bufferChangedBroadcaster.addChangeListener(this);
	display = std::make_unique<TimelineDisplayGL>(processor);
	addAndMakeVisible(*display);
}

TimelineDisplayComponent::~TimelineDisplayComponent()
{
	audioProcessor.bufferChangedBroadcaster.removeChangeListener(this);
	display.reset();
}

void TimelineDisplayComponent::changeListenerCallback(ChangeBroadcaster* source)
{
	auto buffer = audioProcessor.getConvolutionBuffer();
	if(buffer != nullptr)
	{
		display->setInputBuffer(*buffer, audioProcessor.getSampleRate());
	}
}

void TimelineDisplayComponent::visibilityChanged()
{
}

void TimelineDisplayComponent::paint(Graphics& g)
{
	auto bounds = getLocalBounds();
	auto displayBounds = bounds.reduced(60, 30);

	g.setColour(Colours::white);
	g.drawRect(displayBounds.expanded(1, 1), 1);

	for(int i = 5; i >= -45; i -= 5)
	{
		int heightt = (1.0f - (float)(i + 48.0f) / (6.0f + 48.0f)) * displayBounds.getHeight() + 15;
		g.drawFittedText(String(i) + " dB", 15, heightt, 30, 20, Justification::centredRight, 0);
	}

	for(int i = 0; i < displayBounds.getWidth(); i+=100)
	{
		float normalized = (float)i / getWidth();
		float seconds = normalized * (float) display->getBufferSizeUsed() / 44100.0f;
		g.drawFittedText(String(seconds, 2), i + displayBounds.getX(), displayBounds.getBottom() + 7, 30, 20, Justification::centredTop, 0);
	}

}

void TimelineDisplayComponent::resized()
{
	auto bounds = getLocalBounds();

	display->setBounds(bounds.reduced(60, 30));
}
