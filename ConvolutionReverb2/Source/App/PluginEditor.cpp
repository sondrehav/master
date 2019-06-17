/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
ConvolutionReverbAudioProcessorEditor::ConvolutionReverbAudioProcessorEditor (ConvolutionReverbAudioProcessor& p)
    : AudioProcessorEditor (&p), processor (p), domain(p), waveForm(1, p)
{
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
	setResizable(true, true);
    setSize (800, 600);
	setResizeLimits(400, 300, 8000, 6000);
		
	openFileButton.setButtonText("Open file");
	AudioSampleBuffer fileBuffer;
	openFileButton.onClick = [&]()
	{
		FileChooser chooser("Select domain...", {}, "*.*");
		if (chooser.browseForFileToOpen())
		{
			Image img = ImageFileFormat::loadFrom(chooser.getResult());
			Image::BitmapData bitmapData = Image::BitmapData(img, Image::BitmapData::readOnly);
			Image::PixelFormat pixelFormat = bitmapData.pixelFormat;
<<<<<<< Updated upstream
			
			size_t numChannels;
			size_t offset = 1;
			switch(pixelFormat) {
			case Image::ARGB: numChannels = 4; break;
			case Image::RGB: numChannels = 3; break;
			case Image::SingleChannel: numChannels = 1; offset = 0; break;
			default: {
				Logger::writeToLog("Unknown pixel format!");
				return;
			}; }
			float* data = new float[bitmapData.width * bitmapData.height];
			for(int i = 0; i < bitmapData.height; i++)
			{
				for (int j = 0; j < bitmapData.width; j++)
				{
					data[bitmapData.width * i + j] = ((float) bitmapData.data[numChannels*(bitmapData.width*i + j) + offset] / 255.0f); // red channel
				}
			}

			p.setImageData(data, bitmapData.width, bitmapData.height);
				
=======
			if(pixelFormat == Image::ARGB)
			{
				float* data = new float[bitmapData.width * bitmapData.height];
				for(int i = 0; i < bitmapData.height; i++)
				{
					for (int j = 0; j < bitmapData.width; j++)
					{
						data[bitmapData.width * i + j] = (float) bitmapData.data[4*(bitmapData.width*i + j) + 1] / 255.0f; // red channel
					}
				}
				p.setImageData(data, bitmapData.width, bitmapData.height);
				
			}
>>>>>>> Stashed changes
		}
	};

	simulateButton.onClick = [&]()
	{
		processor.simulate(1, 
			[](float p){ Logger::writeToLog(String((int)(p * 100)) + "% ..."); }, 
			[]()
		{
			Logger::writeToLog("Done simulating!");
		}, []() {
			Logger::writeToLog("Cancelled simulation!");
		});
	};

	 xPositionSlider.setName("x");
	 yPositionSlider.setName("y");
	     thetaSlider.setName("r");
	       drySlider.setName("dry");
	       wetSlider.setName("wet");
	   stretchSlider.setName("stretch");
	     pitchSlider.setName("pitch");
	   lowPassSlider.setName("lp");
	  highPassSlider.setName("hp");
    separationSlider.setName("stereo");

	addAndMakeVisible(mainFlex);
	mainFlex.addItem(FlexItem(knobsLeftFlex).withWidth(90));
	mainFlex.addItem(FlexItem(midFlex).withFlex(1,1));
	mainFlex.addItem(FlexItem(knobsRightFlex).withWidth(90));

	knobsLeftFlex.addItem(FlexItem(xPositionSlider).withWidth(80).withHeight(80));
	knobsLeftFlex.addItem(FlexItem(yPositionSlider).withWidth(80).withHeight(80));
	knobsLeftFlex.addItem(FlexItem(thetaSlider).withWidth(80).withHeight(80));
	knobsLeftFlex.addItem(FlexItem(drySlider).withWidth(80).withHeight(80));
	knobsLeftFlex.addItem(FlexItem(wetSlider).withWidth(80).withHeight(80));

	midFlex.addItem(FlexItem(domain).withFlex(7));
	midFlex.addItem(FlexItem(bottomControlsFlex).withFlex(1));

	knobsRightFlex.addItem(FlexItem(stretchSlider).withWidth(80).withHeight(80));
	knobsRightFlex.addItem(FlexItem(pitchSlider).withWidth(80).withHeight(80));
	knobsRightFlex.addItem(FlexItem(lowPassSlider).withWidth(80).withHeight(80));
	knobsRightFlex.addItem(FlexItem(highPassSlider).withWidth(80).withHeight(80));
	knobsRightFlex.addItem(FlexItem(separationSlider).withWidth(80).withHeight(80));

	bottomControlsFlex.addItem(FlexItem(waveForm).withFlex(1, 1));
	bottomControlsFlex.addItem(FlexItem(saveLoadFlex).withWidth(70));
	bottomControlsFlex.addItem(FlexItem(simulateButton).withWidth(70));

	saveLoadFlex.addItem(FlexItem(openFileButton).withFlex(1));
	saveLoadFlex.addItem(FlexItem(saveFileButton).withFlex(1));
	
	mainFlex.flex.flexDirection = FlexBox::Direction::row;
	mainFlex.flex.justifyContent = FlexBox::JustifyContent::spaceBetween;
	mainFlex.flex.alignItems = FlexBox::AlignItems::stretch;
	mainFlex.flex.flexWrap = FlexBox::Wrap::noWrap;

	knobsLeftFlex.flex.flexDirection = FlexBox::Direction::column;
	knobsLeftFlex.flex.justifyContent = FlexBox::JustifyContent::spaceAround;
	knobsLeftFlex.flex.alignItems = FlexBox::AlignItems::center;
	knobsLeftFlex.flex.alignContent = FlexBox::AlignContent::center;
	knobsLeftFlex.flex.flexWrap = FlexBox::Wrap::noWrap;

	midFlex.flex.flexDirection = FlexBox::Direction::column;
	midFlex.flex.justifyContent = FlexBox::JustifyContent::spaceBetween;
	midFlex.flex.alignItems = FlexBox::AlignItems::stretch;
	midFlex.flex.flexWrap = FlexBox::Wrap::noWrap;
	
	knobsRightFlex.flex.flexDirection = FlexBox::Direction::column;
	knobsRightFlex.flex.justifyContent = FlexBox::JustifyContent::spaceAround;
	knobsRightFlex.flex.alignItems = FlexBox::AlignItems::center;
	knobsRightFlex.flex.alignContent = FlexBox::AlignContent::center;
	knobsRightFlex.flex.flexWrap = FlexBox::Wrap::noWrap;

	bottomControlsFlex.flex.flexDirection = FlexBox::Direction::row;
	bottomControlsFlex.flex.justifyContent = FlexBox::JustifyContent::spaceBetween;
	bottomControlsFlex.flex.alignItems = FlexBox::AlignItems::stretch;
	bottomControlsFlex.flex.flexWrap = FlexBox::Wrap::noWrap;

	saveLoadFlex.flex.flexDirection = FlexBox::Direction::column;
	saveLoadFlex.flex.justifyContent = FlexBox::JustifyContent::spaceBetween;
	saveLoadFlex.flex.alignItems = FlexBox::AlignItems::center;
	saveLoadFlex.flex.flexWrap = FlexBox::Wrap::noWrap;



}

ConvolutionReverbAudioProcessorEditor::~ConvolutionReverbAudioProcessorEditor()
{
}

//==============================================================================
void ConvolutionReverbAudioProcessorEditor::paint (Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (ResizableWindow::backgroundColourId));
}

void ConvolutionReverbAudioProcessorEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
	mainFlex.setBounds(getLocalBounds());
	//testSlider.setBounds(Rectangle<int>(100, 100));
}
