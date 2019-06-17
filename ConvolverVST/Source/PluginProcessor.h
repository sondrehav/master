/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#pragma once

#include "../JuceLibraryCode/JuceHeader.h"
#include "Conv/TwoStageFFTConvolver.h"
#include "SimulatorProcessor.h"

//==============================================================================
/**
*/
class ConvolutionReverbAudioProcessor  : public AudioProcessor
{
public:
    //==============================================================================
    ConvolutionReverbAudioProcessor();
    ~ConvolutionReverbAudioProcessor();

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif

    void processBlock (AudioBuffer<float>&, MidiBuffer&) override;

    //==============================================================================
    AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const String getProgramName (int index) override;
    void changeProgramName (int index, const String& newName) override;

    //==============================================================================
    void getStateInformation (MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

	void setConvolutionBufferFromFile(const String& file);

	ChangeBroadcaster bufferChangedBroadcaster;

	inline AudioBuffer<float>* getConvolutionBuffer() { return convolution; }

	float mAccumulatedInputValue = -48.0;
	float mAccumulatedOutputValue = -48.0;

	AudioProcessorValueTreeState& getState() { return state; }


    SimulatorProcessor& getSimulator()
    {
	    return simulator;
    }

private:
	AudioBuffer<float>* convolution = nullptr;
	fftconvolver::TwoStageFFTConvolver* convolvers = nullptr;

	AudioFormatManager formatManager;
	
	AudioProcessorValueTreeState state;

	juce::CriticalSection lock;

	SimulatorProcessor simulator;

	void setConvolutionBuffer(AudioSampleBuffer* buf);


    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ConvolutionReverbAudioProcessor)
};
