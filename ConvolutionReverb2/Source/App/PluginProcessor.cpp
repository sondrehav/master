/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
ConvolutionReverbAudioProcessor::ConvolutionReverbAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", AudioChannelSet::stereo(), true)
                     #endif
                       )
#endif
	, state(*this, nullptr)
{
	formatManager.registerBasicFormats();
	state.state = ValueTree(String("ConvolutionReverbState"));

	solver = new compute::Solver(44100, glm::vec2(-10, -10), glm::vec2(10, 10), glm::vec2(0.2));
	solver->addSource(compute::Source(glm::vec2(-5, 0)));
	solver->addDestination(compute::Destination(glm::vec2(5, 0)));

	size_t width, height;
	solver->getDimensions(&width, &height);
	Logger::writeToLog(String(width) + " x " + String(height));

}

ConvolutionReverbAudioProcessor::~ConvolutionReverbAudioProcessor()
{
	if (this->buffer != nullptr) {
		delete this->buffer;
		this->buffer = nullptr;
	}
	delete solver;
}

//==============================================================================
const String ConvolutionReverbAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool ConvolutionReverbAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool ConvolutionReverbAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool ConvolutionReverbAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double ConvolutionReverbAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int ConvolutionReverbAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int ConvolutionReverbAudioProcessor::getCurrentProgram()
{
    return 0;
}

void ConvolutionReverbAudioProcessor::setCurrentProgram (int index)
{
}

const String ConvolutionReverbAudioProcessor::getProgramName (int index)
{
    return {};
}

void ConvolutionReverbAudioProcessor::changeProgramName (int index, const String& newName)
{
}

//==============================================================================
void ConvolutionReverbAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
	lock.enter();
    // Use this method as the place to do any pre-playback
    // initialisation that you need..
	if(convolvers != nullptr)
	{
		convolvers[0].reset();
		convolvers[1].reset();
	}
	lock.exit();
}

void ConvolutionReverbAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool ConvolutionReverbAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    if (layouts.getMainOutputChannelSet() != AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif

void ConvolutionReverbAudioProcessor::processBlock (AudioBuffer<float>& buffer, MidiBuffer& midiMessages)
{
	lock.enter();

    ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = jmin(getTotalNumInputChannels(), 2);
    auto totalNumOutputChannels = jmin(getTotalNumOutputChannels(), 2);

    // In case we have more outputs than inputs, this code clears any output
    // channels that didn't contain input data, (because these aren't
    // guaranteed to be empty - they may contain garbage).
    // This is here to avoid people getting screaming feedback
    // when they first compile a plugin, but obviously you don't need to keep
    // this code if your algorithm always overwrites all the output channels.
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    // This is the place where you'd normally do the guts of your plugin's
    // audio processing...
    // Make sure to reset the state if your inner loop is processing
    // the samples and the outer loop is handling the channels.
    // Alternatively, you can process the samples with the channels
    // interleaved by keeping the same state.
	int numSamples = buffer.getNumSamples();

	float* buf = new float[numSamples];
    for (int channel = 0; channel < totalNumInputChannels; ++channel)
    {
		auto* readPointer = buffer.getReadPointer(channel);
		if(convolvers != nullptr)
		{
			convolvers[channel].process(readPointer, buf, buffer.getNumSamples());
			buffer.addFrom(channel, 0, buf, numSamples);
		}
        // ..do something to the data...
    }
	delete [] buf;
	lock.exit();
}

//==============================================================================
bool ConvolutionReverbAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

AudioProcessorEditor* ConvolutionReverbAudioProcessor::createEditor()
{
    return new ConvolutionReverbAudioProcessorEditor (*this);
}

//==============================================================================
void ConvolutionReverbAudioProcessor::getStateInformation (MemoryBlock& destData)
{
	std::unique_ptr<XmlElement> xmlState(state.copyState().createXml());
	if (xmlState.get() != nullptr)
		copyXmlToBinary(*xmlState, destData);

}

void ConvolutionReverbAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));

	if (xmlState.get() != nullptr) {
		if (xmlState->hasTagName(state.state.getType()))
		{
			state.replaceState(ValueTree::fromXml(*xmlState));
		}
	}

}

//==============================================================================
// This creates new instances of the plugin..
AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ConvolutionReverbAudioProcessor();
}

void ConvolutionReverbAudioProcessor::setAudioBuffer(const String& filePath)
{
	
	File file(filePath);
	AudioFormatReader* reader = formatManager.createReaderFor(file);
	if (reader != nullptr)
	{
		AudioSampleBuffer* newbuffer = new AudioSampleBuffer(reader->numChannels, reader->lengthInSamples);
		reader->read(newbuffer, 0, reader->lengthInSamples, 0, true, true);
		delete reader;
		setConvolutionBuffer(newbuffer);
		state.state.setProperty("IRFilePath", filePath, nullptr);
	}

}

void ConvolutionReverbAudioProcessor::setConvolutionBuffer(AudioSampleBuffer* buf)
{
	lock.enter();
	if (this->convolvers != nullptr) delete[] this->convolvers;
	if (this->buffer != nullptr) delete this->buffer;
	convolvers = new fftconvolver::TwoStageFFTConvolver[2];
	this->buffer = buf;
	for (int channel = 0; channel < 2; channel++)
	{
		convolvers[channel].init(128, 1024, buffer->getReadPointer(jmin(buffer->getNumChannels() - 1, channel)), buffer->getNumSamples());
	}
	bufferChangedBroadcaster.sendChangeMessage();
	lock.exit();
}
