#include "main.h"
#include <cudaGL.h>
#include "cudaDebug.h"
#include "debug.h"
#include <glm/gtc/matrix_transform.hpp>
#include "../imgui/imgui.h"
#include "../imgui/imgui_helper.h"
#include "../imgui/imgui_impl_gl.h"
#include "helper_math.h"

void WaveSolver::init()
{

	ImGuiH::Init(windowWidth, windowHeight, this);
	ImGui::InitGL();

	initCUDA();
	initCUDAConstants(); 

	uvMatrix = glm::scale(glm::mat4(1.0), glm::vec3((float)width / textureWidth, (float)height / textureHeight, 1.0));
	uvMatrix = glm::translate(uvMatrix, glm::vec3((float)pmlLayers / width, (float)pmlLayers / height, 0.0));

}

void WaveSolver::loop(float dt)
{

	if(simulate)
	{
		simulationLoop();
	}

	if(paintOperation == NonePaintOperation)
	{
		renderer->render(pressureTexture.get(), uvMatrix);
	} else
	{
		renderer->render(geometryTexture.get());
	}

	editorRenderer->render(brushState, paintOperation, initialLineLocation, mousePosition, brushSize, windowWidth, windowHeight);
	
	// ...Draw the UI
	renderUI(windowWidth, windowHeight, dt);

}

void WaveSolver::destroy()
{
	pressureTexture.reset();
	velocityTexture.reset();
	forcesTexture.reset();
	geometryTexture.reset();
	renderer.reset();
	editorRenderer.reset();
	delete inputFile;
}

void WaveSolver::onKeyUp(int keyNum, int mods)
{
	if(keyNum == GLFW_KEY_ESCAPE)
	{
		closeWindow();
	}
	else if (keyNum == GLFW_KEY_LEFT_CONTROL && paintOperation == Add)
	{
		paintOperation = NonePaintOperation;
	}
	else if (keyNum == GLFW_KEY_LEFT_SHIFT && paintOperation == Subtract)
	{
		paintOperation = NonePaintOperation;
	}
}


void WaveSolver::onResized(int width, int height)
{
	GL(glViewport(0, 0, windowWidth, windowHeight));
	auto &imgui_io = ImGui::GetIO();
	imgui_io.DisplaySize = ImVec2(float(windowWidth), float(windowHeight));
}

bool WaveSolver::initCUDA()
{
	CUDA_D(cuInit(0));

	char name[128];
	CUdevice tempDevice;
	CUDA_D(cuDeviceGet(&tempDevice, 0));
	CUDA_D(cuDeviceGetName(name, 128, tempDevice));
	
	DEBUG_STR("Using CUDA device: ", name);
	
	CUDA_D(cuDeviceGet(&device, tempDevice));
	CUDA_D(cuGLCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
	
	CUDA_D(cuModuleLoad(&module, "ptx/test.ptx"));
	CUDA_D(cuModuleGetFunction(&iteratePressureFunction, module, "iteratePressure"));
	CUDA_D(cuModuleGetFunction(&iterateVelocityFunction, module, "iterateVelocity"));
	CUDA_D(cuModuleGetFunction(&firstIterationVelocityFunction, module, "firstIterationVelocity"));
	CUDA_D(cuModuleGetFunction(&drawLineFunction, module, "drawLine"));
	CUDA_D(cuModuleGetFunction(&sampleAtFunction, module, "sampleAt"));
	CUDA_D(cuModuleGetFunction(&sampleInFunction, module, "sampleIn"));

	GL(glEnable(GL_TEXTURE_2D));
	
	pressureTexture = std::make_unique<RWCUDATexture2D>(textureWidth, textureHeight, GL_R32F, GL_CLAMP_TO_BORDER);
	velocityTexture = std::make_unique<RWCUDATexture2D>(textureWidth, textureHeight, GL_R32F, GL_CLAMP_TO_BORDER);
	forcesTexture = std::make_unique<RWCUDATexture2D>(width, height, GL_R32F, GL_CLAMP_TO_BORDER);
	geometryTexture = std::make_unique<RWCUDATexture2D>(width, height, GL_R32F, GL_CLAMP_TO_BORDER);

	//pressureTexture->setFiltering(GL_NEAREST);

	CUDA_D(cuModuleGetSurfRef(&pressureSurfRef, module, "pressureSurfRef"));
	CUDA_D(cuModuleGetTexRef(&pressureTexRef, module, "pressureTexRef"));

	CUDA_D(cuModuleGetSurfRef(&velocitySurfRef, module, "velocitySurfRef"));
	CUDA_D(cuModuleGetTexRef(&velocityTexRef, module, "velocityTexRef"));

	CUDA_D(cuModuleGetSurfRef(&forcesSurfRef, module, "forcesSurfRef"));
	CUDA_D(cuModuleGetTexRef(&forcesTexRef, module, "forcesTexRef"));

	CUDA_D(cuModuleGetTexRef(&geometryTexRef, module, "geometryTexRef"));
	CUDA_D(cuModuleGetSurfRef(&geometrySurfRef, module, "geometrySurfRef"));

	float border[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

	CUDA_D(cuTexRefSetFilterMode(pressureTexRef, CUfilter_mode::CU_TR_FILTER_MODE_LINEAR));
	CUDA_D(cuTexRefSetAddressMode(pressureTexRef, 0, CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER));
	CUDA_D(cuTexRefSetAddressMode(pressureTexRef, 1, CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER));
	CUDA_D(cuTexRefSetBorderColor(pressureTexRef, border));

	CUDA_D(cuTexRefSetFilterMode(velocityTexRef, CUfilter_mode::CU_TR_FILTER_MODE_LINEAR));
	CUDA_D(cuTexRefSetAddressMode(velocityTexRef, 0, CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER));
	CUDA_D(cuTexRefSetAddressMode(velocityTexRef, 1, CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER));
	CUDA_D(cuTexRefSetBorderColor(velocityTexRef, border));

	CUDA_D(cuTexRefSetFilterMode(forcesTexRef, CUfilter_mode::CU_TR_FILTER_MODE_LINEAR));
	CUDA_D(cuTexRefSetAddressMode(forcesTexRef, 0, CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER));
	CUDA_D(cuTexRefSetAddressMode(forcesTexRef, 1, CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER));
	CUDA_D(cuTexRefSetBorderColor(forcesTexRef, border));

	CUDA_D(cuTexRefSetFilterMode(geometryTexRef, CUfilter_mode::CU_TR_FILTER_MODE_LINEAR));
	CUDA_D(cuTexRefSetAddressMode(geometryTexRef, 0, CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP));
	CUDA_D(cuTexRefSetAddressMode(geometryTexRef, 1, CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP));

	pressureTexture->bindToTextureRef(pressureTexRef);
	velocityTexture->bindToTextureRef(velocityTexRef);
	forcesTexture->bindToTextureRef(forcesTexRef);
	geometryTexture->bindToTextureRef(geometryTexRef);
	
	pressureTexture->bindToSurfaceRef(pressureSurfRef);
	velocityTexture->bindToSurfaceRef(velocitySurfRef);
	forcesTexture->bindToSurfaceRef(forcesSurfRef);
	geometryTexture->bindToSurfaceRef(geometrySurfRef);

	renderer = std::make_unique<TextureRenderer>();
	editorRenderer = std::make_unique<EditorRenderer>();

	dimGrid = dim3((textureWidth + dimBlock.x - 1) / dimBlock.x, (textureHeight + dimBlock.y - 1) / dimBlock.y, 1);

	setMaxFPS(60);
	return true;
}

void WaveSolver::copyConstantToDevice(const std::string& name, const void* value)
{
	CUdeviceptr d_constant;
	size_t d_constantBytes;

	CUDA_D(cuModuleGetGlobal(&d_constant, &d_constantBytes, module, name.c_str()));
	CUDA_D(cuMemcpyHtoD(d_constant, value, d_constantBytes));
	
}

void WaveSolver::initCUDAConstants()
{

	copyConstantToDevice("soundVelocity", reinterpret_cast<const void*>(&soundVelocity));
	copyConstantToDevice("stepSize", reinterpret_cast<const void*>(&stepSize));
	copyConstantToDevice("width", reinterpret_cast<const void*>(&textureWidth));
	copyConstantToDevice("height", reinterpret_cast<const void*>(&textureHeight));
	copyConstantToDevice("pmlLayers", reinterpret_cast<const void*>(&pmlLayers));
	copyConstantToDevice("pmlMax", reinterpret_cast<const void*>(&pmlMax));

	copyConstantToDevice("numInputChannels", reinterpret_cast<const void*>(&numInputChannels));
	copyConstantToDevice("numOutputChannels", reinterpret_cast<const void*>(&numOutputChannels));

}

int main()
{
	WaveSolver waveSolverProgram;
	return waveSolverProgram.run();
}

float* WaveSolver::zeros(int numChannels, int width, int height)
{
	float* data = new float[width * height * numChannels];
	std::fill_n(data, width * height * numChannels, 0.0f);
	return data;
}


void WaveSolver::onKeyDown(int keyNum, int mods)
{
	switch(keyNum)
	{
	case GLFW_KEY_SPACE:
		simulate = !simulate;
		break;
	case GLFW_KEY_LEFT_CONTROL:
		paintOperation = Add;
		break;
	case GLFW_KEY_LEFT_SHIFT:
		paintOperation = Subtract;
		break;
	case GLFW_KEY_R:
		resetSimulation();
		break;
	}
}
