#pragma once
#include "../gl/texture.h"
#include <cuda.h>

class RWCUDATexture2D : public Texture
{
public:

	RWCUDATexture2D(int width, int height, GLenum internalFormat, GLenum wrap = GL_CLAMP_TO_BORDER);
	~RWCUDATexture2D();
	void setData(void* data, GLenum dataFormat, GLenum dataType) override;

	void bindToTextureRef(CUtexref texReadRef);
	void bindToSurfaceRef(CUsurfref surfWriteRef);

	const int width;
	const int height;

private:

	CUarray cudaArray;
	CUgraphicsResource cudaGraphicsResource;

};

class RCUDATexture2D : public Texture
{
public:
	RCUDATexture2D(int width, int height, GLenum internalFormat, GLenum wrap = GL_CLAMP_TO_BORDER);
	~RCUDATexture2D();

	void setData(void* data, GLenum dataFormat, GLenum dataType) override;

	void bindToTextureRef(CUtexref texReadRef);

	const int width;
	const int height;

private:

	CUarray cudaArray;
	CUgraphicsResource cudaGraphicsResource;

};
