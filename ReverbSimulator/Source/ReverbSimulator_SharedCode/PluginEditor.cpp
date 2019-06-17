/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
ReverbSimulatorAudioProcessorEditor::ReverbSimulatorAudioProcessorEditor (ReverbSimulatorAudioProcessor& p)
    : AudioProcessorEditor (&p), processor (p), waveView(p)
{
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    setSize (700, 500);
	addAndMakeVisible(waveView);

}

ReverbSimulatorAudioProcessorEditor::~ReverbSimulatorAudioProcessorEditor()
{
}

//==============================================================================
void ReverbSimulatorAudioProcessorEditor::paint (Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
	g.fillAll(getLookAndFeel().findColour(ResizableWindow::backgroundColourId));
}

void ReverbSimulatorAudioProcessorEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
	Rectangle<int> bounds = getLocalBounds();
	waveView.setBounds(bounds.reduced(10, 10));

}
