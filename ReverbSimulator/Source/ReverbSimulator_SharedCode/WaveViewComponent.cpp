#include "WaveViewComponent.h"
#include <fstream>
#include <cassert>
#include <thread>

//==============================================================================
WaveViewComponent::WaveViewComponent(ReverbSimulatorAudioProcessor& p)
    : processor (p)
{
    setSize (400, 300);
	startTimerHz(10);
	context = new compute::CLContext("NVIDIA CUDA", "Quadro P2000 with Max-Q Design");
	solver = new compute::Solver(256, *context);
}

WaveViewComponent::~WaveViewComponent()
{
	shutdownOpenGL();
	delete solver;
	delete context;
}

//==============================================================================
void WaveViewComponent::paint (Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
	//g.fillAll(getLookAndFeel().findColour(ResizableWindow::backgroundColourId));
	g.setColour(Colours::red);
	g.drawRect(getLocalBounds());
}

void WaveViewComponent::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
	Rectangle<int> bounds = getLocalBounds();

}

#define GL(x) { x; GLint err = glGetError(); if(err != GL_NO_ERROR) { Logger::writeToLog("Error: " + String((int)err) + "; File: " + String(__FILE__) + "; Line: " + String(__LINE__)); abort(); }; }

void WaveViewComponent::initialise()
{
	std::ifstream fragIn("frag.fs");
	std::string fragmentSource((std::istreambuf_iterator<char>(fragIn)), std::istreambuf_iterator<char>());
	std::ifstream vertIn("vert.vs");
	std::string vertexSource((std::istreambuf_iterator<char>(vertIn)), std::istreambuf_iterator<char>());
	glEnable(GL_TEXTURE_2D);

	shader = std::make_unique<OpenGLShaderProgram>(openGLContext);
	assert(shader->addVertexShader(vertexSource));
	assert(shader->addFragmentShader(fragmentSource));
	assert(shader->link());

	GL(openGLContext.extensions.glGenBuffers(1, &vertexBufferId));
	GL(openGLContext.extensions.glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId));
	GL(openGLContext.extensions.glBufferData(GL_ARRAY_BUFFER, sizeof(vertexData),
		(GLvoid*)vertexData, GL_STATIC_DRAW));

	GL(openGLContext.extensions.glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(0)));
	GL(openGLContext.extensions.glEnableVertexAttribArray(0));

	GL(openGLContext.extensions.glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(sizeof(float) * 3)));
	GL(openGLContext.extensions.glEnableVertexAttribArray(1));

	GL(openGLContext.extensions.glBindBuffer(GL_ARRAY_BUFFER, 0));
	
	GL(glEnable(GL_TEXTURE_2D));

	GL(glGenTextures(1, &textureBufferId));
	GL(glBindTexture(GL_TEXTURE_2D, textureBufferId));

	
	size_t size;
	std::shared_ptr<float[]> data = solver->getContents(&size);
	assert((int)size == solver->getWidth() * solver->getHeight());
	
	/*float* arr = new float[64 * 64];
	for(int i = 0; i< 64*64;i++) arr[i] = (float)(i / 64) / 64;*/
	
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

	GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, solver->getWidth(), solver->getHeight(), 0, GL_RED, GL_FLOAT, data.get()));
	//delete[] arr;

	GL(glGenTextures(1, &lutBufferId));
	GL(glBindTexture(GL_TEXTURE_1D, lutBufferId));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

	uint8_t* lutData = new uint8_t[5 * 4];
	for(int i = 0; i < 5; i++)
	{
		lutData[4 * i + 0] = lut[i].getRed();
		lutData[4 * i + 1] = lut[i].getGreen();
		lutData[4 * i + 2] = lut[i].getBlue();
		lutData[4 * i + 3] = lut[i].getAlpha();
	}
	GL(glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 5, 0, GL_RGBA, GL_UNSIGNED_BYTE, lutData));
	delete[] lutData;

	shader->use();
	GL(shader->setUniform("tex", (GLint)0));
	GL(shader->setUniform("lut", (GLint)1));
}


void WaveViewComponent::render()
{
	jassert(OpenGLHelpers::isContextActive());


	std::thread::id this_id = std::this_thread::get_id();
	std::cout << __FILE__ << "(" << __LINE__ << "): thread " << this_id << std::endl;

	OpenGLHelpers::clear(Colours::red);

	//openGLContext.extensions.glActiveTexture(GL_TEXTURE0);
	
	GL(glViewport(0, 0, roundToInt(getWidth()), roundToInt(getHeight())));

	shader->use();

	GL(openGLContext.extensions.glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId));

	GL(openGLContext.extensions.glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (GLvoid*)(0)));
	GL(openGLContext.extensions.glEnableVertexAttribArray(0));

	GL(openGLContext.extensions.glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (GLvoid*)(sizeof(float) * 3)));
	GL(openGLContext.extensions.glEnableVertexAttribArray(1));



	GL(openGLContext.extensions.glActiveTexture(GL_TEXTURE0));
	GL(glBindTexture(GL_TEXTURE_2D, textureBufferId));
	if (1)
	{
		for (int i = 0; i < 100; i++) solver->step();
		size_t size;
		std::shared_ptr<float[]> data = solver->getContents(&size);
		assert((int)size == solver->getWidth() * solver->getHeight());

		GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, solver->getWidth(), solver->getHeight(), GL_RED, GL_FLOAT, data.get()));

	}

	GL(openGLContext.extensions.glActiveTexture(GL_TEXTURE1));
	GL(glBindTexture(GL_TEXTURE_1D, lutBufferId));
	

	GL(glDrawArrays(GL_TRIANGLES, 0, 6));

	GL(openGLContext.extensions.glDisableVertexAttribArray(0));
	GL(openGLContext.extensions.glDisableVertexAttribArray(1));
	GL(openGLContext.extensions.glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void WaveViewComponent::shutdown()
{
	
}

void WaveViewComponent::timerCallback()
{
	updateImage = true;
	
}
