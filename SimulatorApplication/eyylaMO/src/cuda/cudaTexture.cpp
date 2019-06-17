#include "cudaTexture.h"
#include "../cudaDebug.h"
#include <cudaGL.h>
#include <cuda_gl_interop.h>

RWCUDATexture2D::RWCUDATexture2D(int width, int height, GLenum internalFormat, GLenum wrap) : Texture(internalFormat, GL_LINEAR, GL_TEXTURE_2D), width(width), height(height)
{
	GL(glGenTextures(1, &textureId));

	GL(glBindTexture(GL_TEXTURE_2D, textureId));

	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap));
	
	GL(glTexStorage2D(GL_TEXTURE_2D, 1, internalFormat, width, height));

	float color[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	GL(glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color));

	GL(glBindTexture(GL_TEXTURE_2D, 0));

	/*
		In OpenGL a GPU texture resource is referenced by a GLuint
		In CUDA a GPU texture resource is referenced by a CUarray
	 */

	 // We only need to do these gl-cuda bindings once.

	 // Register textures in CUDA
	CUDA_D(cuGraphicsGLRegisterImage(&cudaGraphicsResource, textureId, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST));

	CUDA_D(cuGraphicsMapResources(1, &cudaGraphicsResource, 0));

	// Bind the texture to the cuda array.
	CUDA_D(cuGraphicsSubResourceGetMappedArray(&cudaArray, cudaGraphicsResource, 0, 0));

	CUDA_D(cuGraphicsUnmapResources(1, &cudaGraphicsResource, 0));

}

void RWCUDATexture2D::bindToTextureRef(CUtexref texReadRef)
{
	CUDA_D(cuTexRefSetArray(texReadRef, cudaArray, 0));
}


void RWCUDATexture2D::bindToSurfaceRef(CUsurfref surfWriteRef)
{
	CUDA_D(cuSurfRefSetArray(surfWriteRef, cudaArray, 0));
}

void RWCUDATexture2D::setData(void* data, GLenum dataFormat, GLenum dataType)
{
	GL(glBindTexture(GL_TEXTURE_2D, textureId));
	GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, dataFormat, dataType, data));
}

/*
 *
 *
 *
 *
 */

RCUDATexture2D::RCUDATexture2D(int width, int height, GLenum internalFormat, GLenum wrap) : Texture(internalFormat, GL_LINEAR, GL_TEXTURE_2D), width(width), height(height)
{
	GL(glGenTextures(1, &textureId));

	GL(glBindTexture(GL_TEXTURE_2D, textureId));

	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));

	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap));

	GL(glTexStorage2D(GL_TEXTURE_2D, 1, internalFormat, width, height));
	float color[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	GL(glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color));

	GL(glBindTexture(GL_TEXTURE_2D, 0));

	CUDA_D(cuGraphicsGLRegisterImage(&cudaGraphicsResource, textureId, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY));

	CUDA_D(cuGraphicsMapResources(1, &cudaGraphicsResource, 0));

	// Bind the texture to the cuda array.
	CUDA_D(cuGraphicsSubResourceGetMappedArray(&cudaArray, cudaGraphicsResource, 0, 0));

	CUDA_D(cuGraphicsUnmapResources(1, &cudaGraphicsResource, 0));

}

void RCUDATexture2D::setData(void* data, GLenum dataFormat, GLenum dataType)
{
	GL(glBindTexture(GL_TEXTURE_2D, textureId));
	GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, dataFormat, dataType, data));
}

void RCUDATexture2D::bindToTextureRef(CUtexref texReadRef)
{
	CUDA_D(cuTexRefSetArray(texReadRef, cudaArray, 0));
}

RWCUDATexture2D::~RWCUDATexture2D()
{
}

RCUDATexture2D::~RCUDATexture2D()
{
}
