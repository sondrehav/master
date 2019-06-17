#include <glad/glad.h>
#include "TimelineDisplayGL.h"
#include "debug.h"

TimelineDisplayGL::TimelineDisplayGL(ConvolutionReverbAudioProcessor& processor) : processor(processor)
{
	setOpaque(true);
	context.setMultisamplingEnabled(true);
	context.setRenderer(this);
	context.attachTo(*this);
	context.setContinuousRepainting(false);

	dataL = new float[bufferSize];
	dataR = new float[bufferSize];
}

TimelineDisplayGL::~TimelineDisplayGL()
{
	shutdownOpenGL();
	delete [] dataL;
	delete [] dataR;
}

void TimelineDisplayGL::setInputBuffer(AudioSampleBuffer& in, double sr)
{

	float dbInL = std::max<float>(20 * log10(abs(in.getReadPointer(0)[0])), -48) + 10;
	float dbInR = std::max<float>(20 * log10(abs(in.getReadPointer(1)[0])), -48) + 10;

	size_t numSamples = in.getNumSamples();

	for (int sample = 1; sample < numSamples && sample / datapointsPerSample < bufferSize; sample++)
	{

		float lValue = 20 * log10(abs(in.getReadPointer(0)[sample])) + 10;
		float rValue = 20 * log10(abs(in.getReadPointer(1)[sample])) + 10;

		if (lValue > dbInL) dbInL = std::min<float>(dbInL + 10000.0 / (sr * 10) + (lValue - dbInL) * 10000.0 / (sr * 100), 6.0);
		else dbInL = std::max<float>(dbInL - 10000.0 / (sr * 100), -48.0);

		if (rValue > dbInR) dbInR = std::min<float>(dbInR + 10000.0 / (sr * 10) + (rValue - dbInR) * 10000.0 / (sr * 100), 6.0);
		else dbInR = std::max<float>(dbInR - 10000.0 / (sr * 100), -48.0);

		if (sample % datapointsPerSample == 0)
		{
			dataL[sample / datapointsPerSample] = dbInL;
			dataR[sample / datapointsPerSample] = dbInR;
		}
	}

	bufferSizeUsed = std::min(numSamples / datapointsPerSample, bufferSize);
	context.triggerRepaint();
		
}

const std::string vertexShaderSource = "#version 450 core\n\nlayout (location = 0) in float x;\nlayout (location = 1) in float y;\n\nuniform float scaling_x = 1.0;\n\nvoid main(){\n	float yy = 2 * (y + 48) / (6 + 48) - 1;\n	gl_Position = vec4(2 * x * scaling_x - 1, yy, 0.0, 1.0);\n}\n";
const std::string geometryShaderSource = "#version 450 core\n\nlayout (lines) in;\nlayout (triangle_strip, max_vertices = 4) out;\n\nvoid main(){\n	\n	vec4 v1 = gl_in[0].gl_Position;\n	vec4 v2 = gl_in[1].gl_Position;\n	vec4 vb1 = vec4(v1.x, -1.0f, v1.z, v1.w);\n	vec4 vb2 = vec4(v2.x, -1.0f, v2.z, v2.w);\n\n	gl_Position = v1;\n	EmitVertex();\n	gl_Position = v2;\n	EmitVertex();\n	gl_Position = vb1;\n	EmitVertex();\n	gl_Position = vb2;\n	EmitVertex();\n	EndPrimitive();\n\n}\n";
const std::string fragmentShaderSource = "#version 450 core\n\nlayout (location = 0) out vec4 color;\n\nuniform vec4 waveformColor = vec4(1.0f);\n\nvoid main() {\n	color = waveformColor;\n}";

void TimelineDisplayGL::shutdownOpenGL()
{
	context.detach();
}

void TimelineDisplayGL::newOpenGLContextCreated()
{
	if (gladLoadGL() != 1)
	{
		jassert(false);
	}

	float* xs = new float[bufferSize];
	for (int i = 0; i < bufferSize; i++) xs[i] = (float)(i - 1) / bufferSize;

	rightVertexArray = std::make_unique<VertexArray>([this, xs]()
	{
		rightXBuffer = std::make_unique<VertexBuffer>(sizeof(float) * bufferSize);
		rightXBuffer->subdata(xs, 0, sizeof(float) * bufferSize);
		rightXBuffer->with([&]()
		{
			GL(glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, NULL));
			GL(glEnableVertexAttribArray(0));
		});

		rightYBuffer = std::make_unique<VertexBuffer>(sizeof(float) * bufferSize);
		rightYBuffer->with([&]()
		{
			GL(glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, NULL));
			GL(glEnableVertexAttribArray(1));
		});
	});

	leftVertexArray = std::make_unique<VertexArray>([this, xs]()
	{
		leftXBuffer = std::make_unique<VertexBuffer>(sizeof(float) * bufferSize);
		leftXBuffer->subdata(xs, 0, sizeof(float) * bufferSize);
		leftXBuffer->with([&]()
		{
			GL(glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, NULL));
			GL(glEnableVertexAttribArray(0));
		});

		leftYBuffer = std::make_unique<VertexBuffer>(sizeof(float) * bufferSize);
		leftYBuffer->with([&]()
		{
			GL(glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, NULL));
			GL(glEnableVertexAttribArray(1));
		});
	});

	delete[] xs;

	shader = std::make_unique<Shader>();
	shader->attach(vertexShaderSource, GL_VERTEX_SHADER);
	shader->attach(geometryShaderSource, GL_GEOMETRY_SHADER);
	shader->attach(fragmentShaderSource, GL_FRAGMENT_SHADER);
	shader->link();
	shader->validate();

	GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

	auto conv = processor.getConvolutionBuffer();
	if (conv != nullptr) this->setInputBuffer(*conv, processor.getSampleRate());
}

void TimelineDisplayGL::renderOpenGL()
{
	GL(glViewport(0, 0, getWidth(), getHeight()));
	GL(glClear(GL_COLOR_BUFFER_BIT));

	leftYBuffer->subdata(dataL, 0, sizeof(float) * bufferSizeUsed);
	rightYBuffer->subdata(dataR, 0, sizeof(float) * bufferSizeUsed);

	auto c = Colours::darkgrey;
	GL(glClearColor(c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue(), 1));

	if (bufferSizeUsed > 0) {

		shader->with([this]()
		{
			GL(glUniform1f(shader->getUniformLocation("scaling_x"), (float)bufferSize / bufferSizeUsed));

			GL(glUniform4f(shader->getUniformLocation("waveformColor"), 1.0, 0.0, 0.0, 0.25));
			leftVertexArray->draw(bufferSizeUsed, GL_LINE_STRIP);

			GL(glUniform4f(shader->getUniformLocation("waveformColor"), 0.0, 1.0, 0.0, 0.25));
			rightVertexArray->draw(bufferSizeUsed, GL_LINE_STRIP);
		});
	}
}

void TimelineDisplayGL::openGLContextClosing()
{

	leftVertexArray->with([this]()
	{
		leftXBuffer.reset();
		leftYBuffer.reset();
	});
	leftVertexArray.reset();
	rightVertexArray->with([this]()
	{
		rightXBuffer.reset();
		rightYBuffer.reset();
	});
	rightVertexArray.reset();
	shader.reset();
}

void TimelineDisplayGL::paint(Graphics& g)
{
	context.triggerRepaint();
}
