/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#pragma once

#include "../JuceLibraryCode/JuceHeader.h"
#include "../FFTConvolver-non-uniform/TwoStageFFTConvolver.h"
#include "../Simulator/Simulator.h"

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

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

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

	//==============================================================================
	void setAudioBuffer(const String& file);

	void addAudioBufferChangedListener(ChangeListener* l)
	{
		bufferChangedBroadcaster.addChangeListener(l);
	}

	void removeAudioBufferChangedListener(ChangeListener* l)
	{
		bufferChangedBroadcaster.removeChangeListener(l);
	}

	void simulate(float length, const std::function<void(float)>& progressCallback, const std::function<void()>& completionCallback, const std::function<void()>& cancelCallback)
	{
		solver->simulate(length, progressCallback, [completionCallback, this](compute::Solver::impulseContainer container, size_t bufferLength)
		{
			lock.enter();
			const float* buffer = container[std::pair(source, destination)];
			AudioSampleBuffer* b = new AudioSampleBuffer(1, bufferLength);
			b->copyFrom(0, 0, buffer, bufferLength);
			delete[] buffer;
			this->setConvolutionBuffer(b);
			lock.exit();
			completionCallback();
		}, [cancelCallback, this](compute::Solver::impulseContainer container, size_t bufferLength)
		{
			lock.enter();
			/*const float* buffer = container[std::pair(source, destination)];
			float* const* b = (float* const*)buffer;
			this->setConvolutionBuffer(new AudioSampleBuffer(b, 1, bufferLength));*/
			lock.exit();
			cancelCallback();
		});
	}

	void getSimulationDimensions(size_t* width, size_t* height)
	{
		solver->getDimensions(width, height);
	}

	void getSimulationData(float* target)
	{
		solver->getContents(target);
	}

	AudioSampleBuffer* getIR(int sourceChannel, int destinationChannel)
	{
		return this->buffer;
	}

	void setImageData(float* data, size_t width, size_t height)
	{
		lock.enter();
		solver->setImageData(data, width, height);
		lock.exit();
	}

private:
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ConvolutionReverbAudioProcessor)

	AudioSampleBuffer* buffer = nullptr;

    compute::Source source = compute::Source(glm::vec2(-5,0));
    compute::Destination destination = compute::Destination(glm::vec2(5, 0));

	fftconvolver::TwoStageFFTConvolver* convolvers = nullptr;
	AudioFormatManager formatManager;

	juce::CriticalSection lock;
	
	AudioProcessorValueTreeState state;

	ChangeBroadcaster bufferChangedBroadcaster;

	compute::Solver* solver;

	void setConvolutionBuffer(AudioSampleBuffer* buf);
};
