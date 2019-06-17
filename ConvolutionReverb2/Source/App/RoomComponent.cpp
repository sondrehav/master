#include "RoomComponent.h"

#include <fstream>
#include <thread>

//==============================================================================
RoomComponent::RoomComponent(ConvolutionReverbAudioProcessor& p)
	: processor(p)
{
	setSize(400, 300);
	this->camera.width = getWidth();
	this->camera.height = getHeight();
	p.getSimulationDimensions(&this->width, &this->height);
	this->data = new float[this->width * this->height];
	this->listener = [this](size_t width, size_t height)
	{
		this->width = width;
		this->height = height;
		dataLock.enterWrite();
		delete[] data;
		data = new float[width * height];
		dataLock.exitWrite();
	};
}

RoomComponent::~RoomComponent()
{
	shutdownOpenGL();
	delete[] data;
}

//==============================================================================
void RoomComponent::paint(Graphics& g)
{
	// (Our component is opaque, so we must completely fill the background with a solid colour)
	//g.fillAll(getLookAndFeel().findColour(ResizableWindow::backgroundColourId));
	g.setColour(Colours::red);
	g.drawRect(getLocalBounds());
}


#define ASSERT(x) { if(!x) { Logger::writeToLog("Assertion error: " + String(#x) + "; File: " + String(__FILE__) + "; Line: " + String(__LINE__)); abort(); }; }

#define GL(x) { x; GLint err = glGetError(); if(err != GL_NO_ERROR) { Logger::writeToLog("Error: " + String((int)err) + "; File: " + String(__FILE__) + "; Line: " + String(__LINE__)); abort(); }; }

void RoomComponent::resized()
{
	// This is generally where you'll want to lay out the positions of any
	// subcomponents in your editor..
	Rectangle<int> bounds = getLocalBounds();
	this->camera.width = getWidth();
	this->camera.height = getHeight();
	openGLContext.executeOnGLThread([&](OpenGLContext &openGLContext)
	{
		shader->use();
		glm::mat4 m = camera.projection();
		GL(openGLContext.extensions.glUniformMatrix4fv((GLint)projectionMatrixLocation, 1, true, (const GLfloat*)&m));
	}, false);
}
void RoomComponent::initialise()
{
	std::ifstream fragIn("view.fs");
	std::string fragmentSource((std::istreambuf_iterator<char>(fragIn)), std::istreambuf_iterator<char>());
	std::ifstream vertIn("view.vs");
	std::string vertexSource((std::istreambuf_iterator<char>(vertIn)), std::istreambuf_iterator<char>());

	shader = std::make_unique<OpenGLShaderProgram>(openGLContext);
	ASSERT(shader->addVertexShader(vertexSource));
	ASSERT(shader->addFragmentShader(fragmentSource));
	ASSERT(shader->link());

	std::ifstream gridFragIn("grid.fs");
	std::string gridFragmentSource((std::istreambuf_iterator<char>(gridFragIn)), std::istreambuf_iterator<char>());
	std::ifstream gridVertIn("grid.vs");
	std::string gridVertexSource((std::istreambuf_iterator<char>(gridVertIn)), std::istreambuf_iterator<char>());

	gridShader = std::make_unique<OpenGLShaderProgram>(openGLContext);
	ASSERT(gridShader->addVertexShader(gridVertexSource));
	ASSERT(gridShader->addFragmentShader(gridFragmentSource));
	ASSERT(gridShader->link());

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


	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

	GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RED, GL_FLOAT, NULL));

	GL(glGenTextures(1, &lutTextureId));
	GL(glBindTexture(GL_TEXTURE_1D, lutTextureId));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

	uint8_t* lutData = new uint8_t[5 * 4];
	for (int i = 0; i < 5; i++)
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
	projectionMatrixLocation = openGLContext.extensions.glGetUniformLocation(shader->getProgramID(), "projection");
	assert(projectionMatrixLocation >= 0);
	modelMatrixLocation = openGLContext.extensions.glGetUniformLocation(shader->getProgramID(), "model");
	assert(modelMatrixLocation >= 0);
	glm::mat4 modelMatrix = glm::scale(glm::mat4(1.0), glm::vec3(200, 200, 1.0));
	GL(openGLContext.extensions.glUniformMatrix4fv((GLint)modelMatrixLocation, 1, true, (const GLfloat*)&modelMatrix));

	openGLContext.executeOnGLThread([&](OpenGLContext &openGLContext)
	{
		shader->use();
		glm::mat4 m = camera.projection();
		GL(openGLContext.extensions.glUniformMatrix4fv((GLint)projectionMatrixLocation, 1, true, (const GLfloat*)&m));
	}, false);

}


void RoomComponent::render()
{
	jassert(OpenGLHelpers::isContextActive());

	std::thread::id this_id = std::this_thread::get_id();
	std::cout << __FILE__ << "(" << __LINE__ << "): thread " << this_id << std::endl;

	OpenGLHelpers::clear(Colours::black);

	//openGLContext.extensions.glActiveTexture(GL_TEXTURE0);

	GL(glViewport(0, 0, roundToInt(getWidth()), roundToInt(getHeight())));

	GL(openGLContext.extensions.glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId));

	GL(openGLContext.extensions.glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (GLvoid*)(0)));
	GL(openGLContext.extensions.glEnableVertexAttribArray(0));

	GL(openGLContext.extensions.glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (GLvoid*)(sizeof(float) * 3)));
	GL(openGLContext.extensions.glEnableVertexAttribArray(1));

	GL(openGLContext.extensions.glActiveTexture(GL_TEXTURE0));
	GL(glBindTexture(GL_TEXTURE_2D, textureBufferId));

<<<<<<< Updated upstream
	shader->use();

	dataLock.enterWrite();
	processor.getSimulationData(data);
	GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, data));
=======
	dataLock.enterWrite();
	processor.getSimulationData(data);
	GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, data));
	refreshImage = false;
>>>>>>> Stashed changes
	dataLock.exitWrite();

	GL(openGLContext.extensions.glActiveTexture(GL_TEXTURE1));
	GL(glBindTexture(GL_TEXTURE_1D, lutTextureId));


	GL(glDrawArrays(GL_TRIANGLES, 0, 6));

	GL(openGLContext.extensions.glDisableVertexAttribArray(0));
	GL(openGLContext.extensions.glDisableVertexAttribArray(1));
	GL(openGLContext.extensions.glBindBuffer(GL_ARRAY_BUFFER, 0));

	loadGridTexture();
}

void RoomComponent::loadGridTexture()
{
	Image img = ImageFileFormat::loadFrom(File::getCurrentWorkingDirectory().getChildFile("bgGrid.png"));
	Image::BitmapData bitmapData = Image::BitmapData(img, Image::BitmapData::readOnly);
	Image::PixelFormat pixelFormat = bitmapData.pixelFormat;
	assert(pixelFormat == Image::RGB);

	uint8_t* data = new uint8_t[3 * bitmapData.width * bitmapData.height];

	GL(glGenTextures(1, &gridTextureId));
	GL(glBindTexture(GL_TEXTURE_2D, gridTextureId));

	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));

	GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, bitmapData.width, bitmapData.height, 0, GL_RGB, GL_UNSIGNED_BYTE, data));

	GL(glBindTexture(GL_TEXTURE_2D, 0));
	delete[] data;

}

void RoomComponent::shutdown()
{
	GL(openGLContext.extensions.glDeleteBuffers(1, &vertexBufferId));
	GL(glDeleteTextures(1, &lutTextureId));
	GL(glDeleteTextures(1, &textureBufferId));
}


void RoomComponent::mouseWheelMove(const MouseEvent& event, const MouseWheelDetails& wheel)
{
	camera.zoom *= pow(2, wheel.deltaY);
	openGLContext.executeOnGLThread([&](OpenGLContext &openGLContext)
	{
		shader->use();
		glm::mat4 m = camera.projection();
		GL(openGLContext.extensions.glUniformMatrix4fv((GLint)projectionMatrixLocation, 1, true, (const GLfloat*)&m));
	}, false);
}

void RoomComponent::mouseDrag(const MouseEvent& event)
{
	auto modifiers = ModifierKeys::getCurrentModifiers();
	if (mouseState == dragging)
	{
		Logger::writeToLog("mouseDrag");
		if (!modifiers.isLeftButtonDown()) mouseState = none;
		else {
			glm::vec2 mousePosition = glm::vec2(event.position.getX(), -event.position.getY());
			glm::vec2 diff = (mousePosition - mouseEventLast) / camera.zoom;
			camera.position += diff;
			mouseEventLast = mousePosition;
			openGLContext.executeOnGLThread([&](OpenGLContext &openGLContext)
			{
				shader->use();
				glm::mat4 m = camera.projection();
				GL(openGLContext.extensions.glUniformMatrix4fv((GLint)projectionMatrixLocation, 1, true, (const GLfloat*)&m));
			}, false);
		}
	}
	else if (mouseState == none && modifiers.isLeftButtonDown())
	{
		Logger::writeToLog(" start mouseDrag");
		mouseState = dragging;
		mouseEventLast = glm::vec2(event.position.getX(), -event.position.getY());
	}

}
