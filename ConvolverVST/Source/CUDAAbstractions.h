#pragma once
#include "cudaDebug.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <vector>
#include <functional>

namespace cuda
{

class Texture2D;
class SurfaceReference;

template <typename ... Args>
class Kernel;

class Module
{
public:
	Module(const std::string& path);
	Module(const Module&) = delete;
	Module& operator=(const Module&) = delete;

	~Module();

	template <typename ... Args>
	std::shared_ptr<Kernel<Args...>> makeKernel(const std::string& name)
	{
		CUfunction function;
		CUDA_D(cuModuleGetFunction(&function, module, name.c_str()));
		return std::make_shared<Kernel<Args...>>(function);
	}

	SurfaceReference makeSurfaceReference(const std::string& name);

	void copyConstantToDevice(const std::string& id, const void* value);

	void makeCurrent();

private:
	CUmodule module = 0;
	CUcontext context = 0;
	CUdevice device = 0;

};

template <typename ... Args>
class Kernel
{
	friend class Module;

public:
	Kernel(CUfunction function) : function(function)
	{
	}
	
	CUresult operator()(dim3 grid, dim3 block, size_t sharedMem, Args... args)
	{
		void* argsToKernel[sizeof...(Args)] = { &args... };
		return cuLaunchKernel(function, grid.x, grid.y, grid.z, block.x, block.y, block.z, sharedMem, nullptr, argsToKernel, nullptr);
	}
	
private:
	CUfunction function;

};



class Texture2D
{
	friend class TextureRef2D;
	friend class Module;
	friend class SurfaceReference;

public:

	struct TextureProps
	{
		CUaddress_mode addressModes[3] = { CU_TR_ADDRESS_MODE_BORDER , CU_TR_ADDRESS_MODE_BORDER , CU_TR_ADDRESS_MODE_BORDER };
		CUfilter_mode filterMode = CU_TR_FILTER_MODE_LINEAR;
		float borderColor[4] = { 0, 0, 0, 0 };
		int numChannels = 1;
	};

	Texture2D(size_t width, size_t height, TextureProps props);
	Texture2D(size_t width, size_t height);
	Texture2D(const Texture2D&) = delete;
	Texture2D& operator=(const Texture2D&) = delete;
	~Texture2D();

	std::shared_ptr<float[]> getData();
	void getDataAsync(const std::function<void(std::shared_ptr<float[]>)>& fn);
	void setData(const void*);

	CUtexObject getTexObject();
	CUsurfObject getSurfObject();
	

private:
	void uploadData();
	void initialize();

	static void cudaStreamCallback(CUstream hStream, CUresult status, void* userData);

	std::map<CUstream, std::function<void(std::shared_ptr<float[]>)>> cbMap;

	const size_t width, height;
	CUarray cudaArray;
	std::shared_ptr<float[]> data;
	
	CUtexObject* texObject = nullptr;
	CUsurfObject* surfObject = nullptr;

	TextureProps textureProps;


};

class SurfaceReference
{
public:
	SurfaceReference(CUsurfref);
	~SurfaceReference();

private:
	const CUsurfref ref;
};

}
