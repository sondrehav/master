#include <glad/glad.h>
#include "SimulatorGL.h"
#include "debug.h"
#include "helper/ppm.h"

static const std::string vertSource = "#version 450 core\n\nlayout(location = 0) in vec2 position;\nlayout(location = 1) in vec2 uvcoord;\n\nuniform float z = -0.999;\n\nout vec2 uv;\n\nvoid main()\n{\n	uv = uvcoord;\n    gl_Position = vec4(2 * position, z, 1.0);\n}";
static const std::string fragSource = "#version 450 core\n\nlayout (location = 0) out vec4 color;\n\nin vec2 uv;\nuniform sampler2D tex;\nuniform sampler1D lut;\n\nuniform mat4 transform = mat4(1.0);\nuniform int mode = 0;\n\nvoid main()\n{\n	vec2 samplerLocation = (transform * vec4(uv, 0.0, 1.0)).xy;\n	float value = 0;\n	if(mode == 0) { value = texture(tex, samplerLocation).r; }\n	else if (mode == 1) { value = 1.0 / (1.0 + exp(10.0 * texture(tex, samplerLocation).r)); }\n	color = texture(lut, value);\n}";

static const std::string paintVertSource = "#version 450 core\n\nlayout (location = 0) in vec2 position;\n\nuniform mat4 transform = mat4(1.0);\n\nvoid main()\n{\n	gl_Position = transform * vec4(position, 0.0, 1.0);\n}";
static const std::string paintGeomSource = "#version 450 core\n\nlayout (points) in;\nlayout (triangle_strip, max_vertices = 4) out;\n\nuniform float size;\nuniform mat4 transform = mat4(1.0);\n\nout vec2 uv;\n\nvoid main(){\n	vec4 v1 = gl_in[0].gl_Position;\n\n	gl_Position = v1 + transform * vec4(-size, -size, 0.0, 1.0);\n	uv = vec2(0.0, 0.0);\n	EmitVertex();\n	gl_Position = v1 + transform * vec4(size, -size, 0.0, 1.0);\n	uv = vec2(1.0, 0.0);\n	EmitVertex();\n	gl_Position = v1 + transform * vec4(-size, size, 0.0, 1.0);\n	uv = vec2(0.0, 1.0);\n	EmitVertex();\n	gl_Position = v1 + transform * vec4(size, size, 0.0, 1.0);\n	uv = vec2(1.0, 1.0);\n	EmitVertex();\n	EndPrimitive();\n}";
static const std::string paintFragSource = "#version 450 core\n\nlayout (location = 0) out vec4 color;\n\nin vec2 uv;\n\nuniform float amount;\nuniform float falloff;\n\nvoid main(){\n	float rx = pow(2.0 * uv.x - 1.0, 2.0);\n	float ry = pow(2.0 * uv.y - 1.0, 2.0);\n	float value = min((1.0 - rx - ry) / falloff, 1.0) * amount;\n	color = vec4(1.0, 1.0, 1.0, value);\n}";

SimulatorGLComponent::SimulatorGLComponent(ConvolutionReverbAudioProcessor& audioProcessor) 
	: processor(audioProcessor)
{
	setOpaque(true);
	context.setMultisamplingEnabled(true);
	context.setRenderer(this);
	context.attachTo(*this);
	context.setContinuousRepainting(false);
	
	displayChanged = new ChangeListenerHelper([&](ChangeBroadcaster* src)
	{
		if(context.isAttached())
		{
			const SimulatorProcessor& sim = processor.getSimulator();
			if (sim.getWidth() != width || sim.getHeight() != height) return;
			context.executeOnGLThread([this](const OpenGLContext&)
			{
				auto data = processor.getSimulator().getPressureFieldData();
				texture->setData((void*) data.get(), GL_RED, GL_FLOAT);
			}, false);
		}
	});

	dimensionsChanged = new ChangeListenerHelper([&](ChangeBroadcaster* src)
	{
		if (context.isAttached())
		{
			context.executeOnGLThread([this](const OpenGLContext&)
			{
				width = processor.getSimulator().getWidth();
				height = processor.getSimulator().getHeight();
				uninitializeTextures();
				initializeTextures();
			}, false);
		}
		printf("Dimensions changed :)\n");
	});

	geometryChanged = new ChangeListenerHelper([&](ChangeBroadcaster* src)
	{
		if (context.isAttached())
		{
			context.executeOnGLThread([this](const OpenGLContext&)
			{
				auto data = processor.getSimulator().getGeometryData();
				geometryTexture->setData((void*)data.get(), GL_RED, GL_FLOAT);
			}, false);
		}
		printf("Geometry changed :)\n");
	});

	processor.getSimulator().getOutputFieldChanged().addChangeListener(displayChanged);
	processor.getSimulator().getDimensionsChanged().addChangeListener(dimensionsChanged);
	processor.getSimulator().getGeometryChanged().addChangeListener(geometryChanged);
}

SimulatorGLComponent::~SimulatorGLComponent()
{
	shutdownOpenGL();

	processor.getSimulator().getOutputFieldChanged().removeChangeListener(displayChanged);
	processor.getSimulator().getDimensionsChanged().removeChangeListener(dimensionsChanged);
	processor.getSimulator().getGeometryChanged().removeChangeListener(geometryChanged);

	delete displayChanged;
	delete dimensionsChanged;
	delete geometryChanged;
}

void SimulatorGLComponent::shutdownOpenGL()
{
	context.detach();
}


void SimulatorGLComponent::initializeTextures()
{

	texture = std::make_shared<Texture2D>(width, height, GL_R32F);
	float* data = new float[width * height];
	//for (int i = 0; i < width * height; i++) data[i] = (float)(rand() % RAND_MAX) / RAND_MAX;
	std::fill_n(data, width * height, 0.0f);
	texture->setData(data, GL_RED, GL_FLOAT);
	texture->setWrapping(GL_CLAMP_TO_BORDER, GL_CLAMP_TO_BORDER);
	texture->setFiltering(GL_LINEAR);

	geometryTexture = std::make_shared<Texture2D>(width, height, GL_R32F);
	geometryTexture->setData(data, GL_RED, GL_FLOAT);
	geometryTexture->setWrapping(GL_CLAMP_TO_BORDER, GL_CLAMP_TO_BORDER);
	geometryTexture->setFiltering(GL_LINEAR);

	delete[] data;
}

void SimulatorGLComponent::uninitializeTextures()
{
	texture.reset();
	geometryTexture.reset();
}

void SimulatorGLComponent::newOpenGLContextCreated()
{
	if (gladLoadGL() != 1)
	{
		jassert(false);
	}

	width = processor.getSimulator().getWidth();
	height = processor.getSimulator().getHeight();

	initializeTextures();

	lutTexture = std::make_shared<Texture1D>(5, GL_RGBA8);
	lutTexture->setData((void*)lut, GL_RGBA, GL_UNSIGNED_BYTE);
	lutTexture->setWrapping(GL_CLAMP_TO_BORDER);
	lutTexture->setFiltering(GL_LINEAR);

	geometryLutTexture = std::make_shared<Texture1D>(2, GL_RGBA8);
	geometryLutTexture->setData((void*)geometryLut, GL_RGBA, GL_UNSIGNED_BYTE);
	geometryLutTexture->setWrapping(GL_CLAMP_TO_BORDER);
	geometryLutTexture->setFiltering(GL_LINEAR);

	screenShader = std::make_unique<Shader>();
	screenShader->attach(vertSource, GL_VERTEX_SHADER);
	screenShader->attach(fragSource, GL_FRAGMENT_SHADER);
	screenShader->link();
	screenShader->with([&]()
	{
		GL(glUniform1i(screenShader->getUniformLocation("tex"), 0));
		GL(glUniform1i(screenShader->getUniformLocation("lut"), 1));
	});
	screenShader->validate();

	screenQuadArray = std::make_unique<VertexArray>([&]()
	{
		screenQuadBuffer = std::make_unique<VertexBuffer>(sizeof(vertexData));
		screenQuadBuffer->subdata((void*)vertexData, 0, sizeof(vertexData));
		screenQuadBuffer->with([&]()
		{
			GL(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(0)));
			GL(glEnableVertexAttribArray(0));
			GL(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(sizeof(float) * 3)));
			GL(glEnableVertexAttribArray(1));
		});
	});

	paintingArray = std::make_unique<VertexArray>([&]()
	{
		paintingBuffer = std::make_unique<VertexBuffer>(2 * sizeof(float));
		float p[2] = { 0, 0 };
		paintingBuffer->subdata((void*)p, 0, 2 * sizeof(float));
		paintingBuffer->with([&]()
		{
			GL(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (void*)(0)));
			GL(glEnableVertexAttribArray(0));
		});
	});

	paintingShader = std::make_unique<Shader>();
	paintingShader->attach(paintVertSource, GL_VERTEX_SHADER);
	paintingShader->attach(paintGeomSource, GL_GEOMETRY_SHADER);
	paintingShader->attach(paintFragSource, GL_FRAGMENT_SHADER);
	paintingShader->link();
	paintingShader->validate();
	
}

void SimulatorGLComponent::renderOpenGL()
{
	printf("\nDisplay changed :)\n\n");

	float matrix[16] = { 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1 };

	GL(glViewport(0, 0, getWidth(), getHeight()));
	GL(glClearColor(0, 0, 0, 1));
	GL(glClear(GL_COLOR_BUFFER_BIT));

	GL(glEnable(GL_BLEND));
	GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
	renderPressureField(matrix);
	renderGeometry(matrix);

	renderPaintBrush(matrix);
	
}

void SimulatorGLComponent::renderPressureField(const float* matrix)
{
	float mm[16];
	std::memcpy(mm, matrix, sizeof(float) * 16);
	screenShader->with([this, mm]()
	{
		GL(glUniformMatrix4fv(screenShader->getUniformLocation("transform"), 1, false, (GLfloat*)&mm));
		GL(glUniform1i(screenShader->getUniformLocation("mode"), 1));
		Texture::withMultiple({ texture, lutTexture }, [&]()
		{
			screenQuadArray->draw(6, GL_TRIANGLES);
		});
	});
}

void SimulatorGLComponent::openGLContextClosing()
{
	screenQuadArray->with([&]()
	{
		screenQuadBuffer.reset();
	});
	screenQuadArray.reset();
	texture.reset();
	lutTexture.reset();
	geometryTexture.reset();
	geometryLutTexture.reset();
	screenShader.reset();
	paintingShader.reset();
	paintingArray->with([&]()
	{
		paintingBuffer.reset();
	});
	paintingArray.reset();
}

void SimulatorGLComponent::paint(Graphics& g)
{
	context.triggerRepaint();
}
