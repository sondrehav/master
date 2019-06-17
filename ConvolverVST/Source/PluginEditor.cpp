/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
ConvolutionReverbAudioProcessorEditor::ConvolutionReverbAudioProcessorEditor(ConvolutionReverbAudioProcessor& p)
	: AudioProcessorEditor(&p), processor(p), timelineComponent(p), simulatorComponent(p),
	inputMeter({ -48, 6 }, [&]() { return processor.mAccumulatedInputValue; }),
	outputMeter({ -48, 6 }, [&]() {return processor.mAccumulatedOutputValue; }),
	refreshRateCounter(5),
	iterationCounter(5)
{
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
	setResizable(true, true);
    setSize (1200, 800);
	setResizeLimits(400, 300, 8000, 6000);
	openFile.setButtonText("Load IR");
	openFile.onClick = [&]()
	{
		FileChooser chooser("Select a Wave file to load as impulse response.", {}, "*.wav");
		if (chooser.browseForFileToOpen())
		{
			String file = chooser.getResult().getFullPathName();
			processor.setConvolutionBufferFromFile(file);
		}
	};
	addAndMakeVisible(openFile);
	addAndMakeVisible(timelineComponent);
	addChildComponent(simulatorComponent);
	addAndMakeVisible(inputMeter);
	addAndMakeVisible(outputMeter);
	addAndMakeVisible(wetSlider);
	addAndMakeVisible(drySlider);

	const std::function<String(float)>& mdBMeterLabels = [](float value) {return String(abs(value), 0); };
	inputMeter.setLabels(mdBMeterLabels);
	outputMeter.setLabels(mdBMeterLabels);

	wetSlider.setRange(-48, 6);
	drySlider.setRange(-48, 6);

	wetSlider.setValue(0);
	drySlider.setValue(0);

	wetSlider.setSliderStyle(Slider::RotaryVerticalDrag);
	drySlider.setSliderStyle(Slider::RotaryVerticalDrag);

	wetSlider.setTextBoxStyle(Slider::TextBoxBelow, false, 60, 20);
	drySlider.setTextBoxStyle(Slider::TextBoxBelow, false, 60, 20);

	wetSliderAttachment = std::make_unique<AudioProcessorValueTreeState::SliderAttachment>(processor.getState(), "wet", wetSlider);
	drySliderAttachment = std::make_unique<AudioProcessorValueTreeState::SliderAttachment>(processor.getState(), "dry", drySlider);

	toggleMode.setButtonText("Switch mode");

	addAndMakeVisible(toggleMode);

	toggleMode.onStateChange = [&]()
	{
		bool value = toggleMode.getToggleState();
		timelineComponent.setVisible(!value);
		simulatorComponent.setVisible(value);
	};
	toggleMode.setToggleState(false, dontSendNotification);

	startSimulation.setButtonText("Start");
	startSimulation.onClick = [&]() { processor.getSimulator().start(); };
	addAndMakeVisible(startSimulation);

	stopSimulation.setButtonText("Stop");
	stopSimulation.onClick = [&]() { processor.getSimulator().stop(); };
	addAndMakeVisible(stopSimulation);

	pauseSimulation.setButtonText("Pause");
	pauseSimulation.onClick = [&]() { processor.getSimulator().pause(); };
	addAndMakeVisible(pauseSimulation);

	processor.getSimulator().getSimulatorStateChanged().addChangeListener(this);

	refreshRateCounter.getValueFn = [&]()
	{
		float freq = 1.0f / processor.getSimulator().getSecondsPerIteration();
		return String(freq, 3) + String(" hz");
	};
	addAndMakeVisible(refreshRateCounter);

	iterationCounter.getValueFn = [&]()
	{
		float freq = 1.0f / processor.getSimulator().getSecondsPerIteration();
		return String(processor.getSimulator().getNumIterations());
	};
	addAndMakeVisible(iterationCounter);
	
	dimensionChooser.addItem("100 x 100", 1);
	dimensionChooser.addItem("200 x 200", 2);
	dimensionChooser.addItem("500 x 500", 3);
	dimensionChooser.addItem("1000 x 1000", 4);
	dimensionChooser.addItem("2000 x 2000", 5);
	addAndMakeVisible(dimensionChooser);

	dimensionChooser.onChange = [&]()
	{
		int width, height;
		switch(dimensionChooser.getSelectedId())
		{
		case 1: width = 100; height = 100; break;
		case 2: width = 200; height = 200; break;
		case 3: width = 500; height = 500; break;
		case 4: width = 1000; height = 1000; break;
		case 5: width = 2000; height = 2000; break;
		default: width = 100; height = 100; break;
		}
		processor.getSimulator().setDimensions(width, height);
	};

}

ConvolutionReverbAudioProcessorEditor::~ConvolutionReverbAudioProcessorEditor()
{
	processor.getSimulator().getSimulatorStateChanged().removeChangeListener(this);
}

void ConvolutionReverbAudioProcessorEditor::resized()
{
	auto bounds = getLocalBounds();
	bounds.reduce(5, 5);
	auto bottom = bounds.removeFromBottom(40);
	outputMeter.setBounds(bounds.removeFromRight(20));
	inputMeter.setBounds(bounds.removeFromRight(20));
	
	openFile.setBounds(bottom);

	auto slidersBounds = bounds.removeFromBottom(80);
	wetSlider.setBounds(slidersBounds.removeFromLeft(60));
	drySlider.setBounds(slidersBounds.removeFromLeft(60));
	toggleMode.setBounds(slidersBounds.removeFromLeft(60));
	
	auto l = bounds.removeFromBottom(50);
	startSimulation.setBounds(l.removeFromLeft(60));
	pauseSimulation.setBounds(l.removeFromLeft(60));
	stopSimulation.setBounds(l.removeFromLeft(60));
	dimensionChooser.setBounds(l.removeFromLeft(60));
	refreshRateCounter.setBounds(l.removeFromLeft(140));
	iterationCounter.setBounds(l.removeFromLeft(140));


	timelineComponent.setBounds(bounds);
	simulatorComponent.setBounds(bounds);
}

void ConvolutionReverbAudioProcessorEditor::paint(Graphics& g)
{
	g.setColour(Colours::darkgrey);
	g.fillRect(getLocalBounds());
}

void ConvolutionReverbAudioProcessorEditor::changeListenerCallback(ChangeBroadcaster* source)
{
	switch(processor.getSimulator().getSimulationState())
	{
	case SimulatorProcessor::Started:
		startSimulation.setEnabled(false);
		pauseSimulation.setEnabled(true);
		stopSimulation.setEnabled(true);
		dimensionChooser.setEnabled(false);
		break;
	case SimulatorProcessor::Paused:
		startSimulation.setEnabled(true);
		pauseSimulation.setEnabled(false);
		stopSimulation.setEnabled(true);
		dimensionChooser.setEnabled(false);
		break;
	case SimulatorProcessor::Stopped:
		startSimulation.setEnabled(true);
		pauseSimulation.setEnabled(false);
		stopSimulation.setEnabled(false);
		dimensionChooser.setEnabled(true);
		break;
	}
}
