#include "CUDAAbstractions.h"
#include <glad/glad.h>
#include <cudaGL.h>
#include <tuple>
#include <functional>
#include <deque>
#include <cassert>


namespace cuda {


Module::Module(const std::string& path)
{
	
	CUDA_D(cuInit(0));

	char name[128];
	CUdevice tempDevice;
	CUDA_D(cuDeviceGet(&tempDevice, 0));
	CUDA_D(cuDeviceGetName(name, 128, tempDevice));

    printf("Using CUDA device: %s\n", name);
	CUDA_D(cuDeviceGet(&device, tempDevice));
	
	CUDA_D(cuCtxCreate(&context, 0, device));

	CUDA_D(cuModuleLoad(&module, path.c_str()));

}

Module::~Module()
{
	CUDA_D(cuModuleUnload(module));
}
void Module::makeCurrent()
{
	CUDA_D(cuCtxSetCurrent(context));
}

void Module::copyConstantToDevice(const std::string& id, const void* value)
{

	CUdeviceptr d_constant;
	size_t d_constantBytes;

	CUDA_D(cuModuleGetGlobal(&d_constant, &d_constantBytes, module, id.c_str()));
	CUDA_D(cuMemcpyHtoD(d_constant, value, d_constantBytes));

}

Texture2D::Texture2D(size_t width, size_t height) :
	width(width), height(height)
{
	textureProps = TextureProps();
	data = std::shared_ptr<float[]>(new float[width * height * textureProps.numChannels]);
	initialize();
}

Texture2D::Texture2D(size_t width, size_t height, TextureProps props) : textureProps(props),
	width(width), height(height)
{
	data = std::shared_ptr<float[]>(new float[width * height * textureProps.numChannels]);
	initialize();
}

static std::map<int, std::pair<Texture2D*, std::deque<std::function<void(std::shared_ptr<float[]>)>>>> textureCallbacks;

Texture2D::~Texture2D()
{
	CUDA_D(cuArrayDestroy(cudaArray));
	if(surfObject != nullptr) CUDA_D(cuSurfObjectDestroy(*surfObject));
	if(texObject != nullptr) CUDA_D(cuSurfObjectDestroy(*texObject));
	delete texObject;
	delete surfObject;
}


void Texture2D::cudaStreamCallback(CUstream hStream, CUresult status, void* userData)
{
	CUDA_D(status);
	Texture2D* texture = (Texture2D*) userData;
	texture->cbMap[hStream](texture->data);
	texture->cbMap.erase(hStream);
}

void Texture2D::initialize()
{
	std::memset(data.get(), 0.0f, width * height * textureProps.numChannels * sizeof(float));
	CUDA_ARRAY3D_DESCRIPTOR descriptor;
	descriptor.Depth = 0;
	descriptor.Width = width;
	descriptor.Height = height;
	descriptor.Format = CU_AD_FORMAT_FLOAT;
	descriptor.NumChannels = textureProps.numChannels;
	descriptor.Flags = CUDA_ARRAY3D_SURFACE_LDST;

	CUDA_D(cuArray3DCreate_v2(&cudaArray, &descriptor));
}

void Texture2D::setData(const void* dt)
{
	std::memcpy(data.get(), dt, width * height * textureProps.numChannels * sizeof(float));
	uploadData();
}


void Texture2D::uploadData()
{
	CUDA_MEMCPY2D cpy;

	cpy.WidthInBytes = width * sizeof(float) * textureProps.numChannels;
	cpy.Height = height;
	
	cpy.dstArray = cudaArray;
	cpy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	cpy.dstXInBytes = cpy.dstY = 0;

	cpy.srcHost = data.get();
	cpy.srcXInBytes = cpy.srcY = 0;
	cpy.srcPitch = cpy.WidthInBytes;
	cpy.srcMemoryType = CU_MEMORYTYPE_HOST;

	CUDA_D(cuMemcpy2D_v2(&cpy));
}

std::shared_ptr<float[]> Texture2D::getData()
{
	
	CUDA_MEMCPY2D cpy;

	cpy.WidthInBytes = width * sizeof(float) * textureProps.numChannels;
	cpy.Height = height;

	cpy.srcArray = cudaArray;
	cpy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
	cpy.srcXInBytes = cpy.srcY = 0;

	cpy.dstHost = (void*) data.get();
	cpy.dstXInBytes = cpy.dstY = 0;
	cpy.dstPitch = cpy.WidthInBytes;
	cpy.dstMemoryType = CU_MEMORYTYPE_HOST;

	CUDA_D(cuMemcpy2D_v2(&cpy));
	return data;
}

void Texture2D::getDataAsync(const std::function<void(std::shared_ptr<float[]>)>& fn)
{
	CUDA_MEMCPY2D cpy;

	cpy.WidthInBytes = width * sizeof(float) * textureProps.numChannels;
	cpy.Height = height;

	cpy.srcArray = cudaArray;
	cpy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
	cpy.srcXInBytes = cpy.srcY = 0;

	cpy.dstHost = (void*)data.get();
	cpy.dstXInBytes = cpy.dstY = 0;
	cpy.dstPitch = cpy.WidthInBytes;
	cpy.dstMemoryType = CU_MEMORYTYPE_HOST;

	CUstream memCpyStream;
	CUDA_D(cuStreamCreate(&memCpyStream, CU_STREAM_NON_BLOCKING));
	cbMap[memCpyStream] = fn;
	CUDA_D(cuStreamAddCallback(memCpyStream, cudaStreamCallback, this, 0));
	CUDA_D(cuMemcpy2DAsync_v2(&cpy, memCpyStream));
}

CUtexObject Texture2D::getTexObject()
{
	if(texObject == nullptr) {
		texObject = new CUtexObject;
		CUDA_RESOURCE_DESC resDesc;
		std::memset(&resDesc, 0, sizeof(CUDA_RESOURCE_DESC));
		resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
		resDesc.res.array.hArray = cudaArray;
		

		CUDA_TEXTURE_DESC texDesc;
		std::memset(&texDesc, 0, sizeof(CUDA_TEXTURE_DESC));
		texDesc.addressMode[0] = textureProps.addressModes[0];
		texDesc.addressMode[1] = textureProps.addressModes[1];
		texDesc.addressMode[2] = textureProps.addressModes[2];
		texDesc.filterMode = textureProps.filterMode;
		texDesc.borderColor[0] = textureProps.borderColor[0];
		texDesc.borderColor[1] = textureProps.borderColor[1];
		texDesc.borderColor[2] = textureProps.borderColor[2];
		texDesc.borderColor[3] = textureProps.borderColor[3];
		

		/*CUDA_RESOURCE_VIEW_DESC rwDesc;
		std::memset(&rwDesc, 0, sizeof(CUDA_RESOURCE_VIEW_DESC));
		rwDesc.format = CU_RES_VIEW_FORMAT_FLOAT_1X32;
		rwDesc.width = width;
		rwDesc.height = height;
		rwDesc.depth = 1;*/

		CUDA_D(cuTexObjectCreate(texObject, &resDesc, &texDesc, NULL));
	}
	return *texObject;

}

CUsurfObject Texture2D::getSurfObject()
{
	if (surfObject == nullptr) {
		surfObject = new CUsurfObject;
		CUDA_RESOURCE_DESC resDesc;
		std::memset(&resDesc, 0, sizeof(CUDA_RESOURCE_DESC));
		resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
		resDesc.res.array.hArray = cudaArray;
		CUDA_D(cuSurfObjectCreate(surfObject, &resDesc));
	}
	return *surfObject;
}



SurfaceReference Module::makeSurfaceReference(const std::string& name)
{
	CUsurfref ref;
	CUDA_D(cuModuleGetSurfRef(&ref, module, name.c_str()));
	return SurfaceReference(ref);
}

SurfaceReference::SurfaceReference(CUsurfref ref) : ref(ref)
{

}

SurfaceReference::~SurfaceReference() {}



}
