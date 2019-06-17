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
, state(*this, nullptr),
	simulator(200, 200, getSampleRate())
{
	formatManager.registerBasicFormats();
	auto gainString = [](float value) {return value <= -48.0 ? "-inf" : String(value, 2) + " dB"; };
	state.createAndAddParameter("wet", "Wet", "Wet", NormalisableRange<float>(-48.0, 6.0), 0.0, gainString, nullptr);
	state.createAndAddParameter("dry", "Dry", "Dry", NormalisableRange<float>(-48.0, 6.0), 0.0, gainString, nullptr);
	state.state = ValueTree(String("ConvolutionReverbState"));
	simulator.setSampleRate(getSampleRate());
}

ConvolutionReverbAudioProcessor::~ConvolutionReverbAudioProcessor()
{
	delete[] convolvers;
	delete convolution;
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
    // Use this method as the place to do any pre-playback
    // initialisation that you need..
	simulator.setSampleRate(getSampleRate());
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
	auto totalNumInputChannels = jmin(getTotalNumInputChannels(), 2);
	auto totalNumOutputChannels = jmin(getTotalNumOutputChannels(), 2);

	// In case we have more outputs than inputs, this code clears any output
	// channels that didn't contain input data, (because these aren't
	// guaranteed to be empty - they may contain garbage).
	// This is here to avoid people getting screaming feedback
	// when they first compile a plugin, but obviously you don't need to keep
	// this code if your algorithm always overwrites all the output channels.
	for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
		buffer.clear(i, 0, buffer.getNumSamples());

	// This is the place where you'd normally do the guts of your plugin's
	// audio processing...
	// Make sure to reset the state if your inner loop is processing
	// the samples and the outer loop is handling the channels.
	// Alternatively, you can process the samples with the channels
	// interleaved by keeping the same state.

	AudioSampleBuffer b(totalNumInputChannels, buffer.getNumSamples());
	if (convolvers != nullptr)
	{
		for (int channel = 0; channel < totalNumInputChannels; ++channel)
		{
			auto* readPointer = buffer.getReadPointer(channel);
			convolvers[channel].process(readPointer, b.getWritePointer(channel), b.getNumSamples());
		}
	}

	for (int sample = 0; sample < buffer.getNumSamples(); sample++)
	{

		float dbIn = -48.0;
		float dbOut = -48.0;

		float dry = *state.getRawParameterValue("dry");
		float wet = *state.getRawParameterValue("wet");

		float dryValue = pow(10, dry / 20);
		float wetValue = pow(10, wet / 20);

		for (int channel = 0; channel < std::min<int>(totalNumInputChannels, 2); ++channel)
		{
			auto* channelData = buffer.getReadPointer(channel);
			auto* writeChannelData = buffer.getWritePointer(channel);

			float wet = 0;
			float dry = channelData[sample] * dryValue;

			if(convolvers != nullptr)
			{
				auto* wetData = b.getReadPointer(channel);
				wet = wetData[sample] * wetValue * pow(10, -29.21 / 20);
			}

			float out = wet + dry;
			writeChannelData[sample] = out;

			dbIn = std::max<float>(20 * log10(abs(dry)), dbIn);
			dbOut = std::max<float>(20 * log10(abs(out)), dbOut);

		}
		if (dbIn> mAccumulatedInputValue) mAccumulatedInputValue = dbIn;
		else mAccumulatedInputValue = std::max<float>(mAccumulatedInputValue - 10000.0 / (getSampleRate() * 100), -48.0);
		if (dbOut > mAccumulatedOutputValue) mAccumulatedOutputValue = dbOut;
		else mAccumulatedOutputValue = std::max<float>(mAccumulatedOutputValue - 10000.0 / (getSampleRate() * 100), -48.0);
		
	}
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
void ConvolutionReverbAudioProcessor::getStateInformation(MemoryBlock& destData)
{
	// Store an xml representation of our state.
	std::unique_ptr<XmlElement> xmlState(state.copyState().createXml());

	if (xmlState.get() != nullptr)
		copyXmlToBinary(*xmlState, destData);
}

void ConvolutionReverbAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
	// Restore our plug-in's state from the xml representation stored in the above
	// method.
	std::unique_ptr<XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));

	if (xmlState.get() != nullptr) {
		// ok, we want to load our processor state from the value tree that we have saved
		if (xmlState->hasTagName(state.state.getType()))
		{
			state.replaceState(ValueTree::fromXml(*xmlState));
			String path = state.state.getProperty("IRFilePath");
			if(path != "")
			{
				setConvolutionBufferFromFile(path);
			}
		}
	}
}

//==============================================================================
// This creates new instances of the plugin..
AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ConvolutionReverbAudioProcessor();
}



void ConvolutionReverbAudioProcessor::setConvolutionBufferFromFile(const String& filePath)
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
		bufferChangedBroadcaster.sendChangeMessage();
	}
}

void ConvolutionReverbAudioProcessor::setConvolutionBuffer(AudioSampleBuffer* buf)
{
	lock.enter();
	if (this->convolvers != nullptr) delete[] this->convolvers;
	if (this->convolution != nullptr) delete this->convolution;
	convolvers = new fftconvolver::TwoStageFFTConvolver[2];
	this->convolution = buf;
	if(this->convolution->getNumChannels() <= 1)
	{
		AudioSampleBuffer* newBuffer = new AudioSampleBuffer(2, this->convolution->getNumSamples());
		std::memset(newBuffer->getWritePointer(0), 0, newBuffer->getNumSamples() * sizeof(float));
		std::memset(newBuffer->getWritePointer(1), 0, newBuffer->getNumSamples() * sizeof(float));
		newBuffer->addFrom(0, 0, *this->convolution, 0, 0, this->convolution->getNumSamples());
		newBuffer->addFrom(1, 0, *this->convolution, 0, 0, this->convolution->getNumSamples());
		delete convolution;
		this->convolution = newBuffer;
	}
	for (int channel = 0; channel < 2; channel++)
	{
		convolvers[channel].init(128, 1024, convolution->getReadPointer(jmin(convolution->getNumChannels() - 1, channel)), convolution->getNumSamples());
	}
	lock.exit();
}
