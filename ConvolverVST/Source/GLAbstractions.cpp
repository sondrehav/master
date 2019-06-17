#include <glad/glad.h>
#include "GLAbstractions.h"
#include "debug.h"
#include <cassert>

VertexBuffer::VertexBuffer(size_t size) : size(size)
{
	data = new uint8_t[size];
	std::memset(data, 0, size);

	GL(glGenBuffers(1, &vbo));
	GL(glBindBuffer(GL_ARRAY_BUFFER, vbo));
	GL(glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW));
	GL(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

VertexBuffer::~VertexBuffer()
{
	uint8_t* dt = (uint8_t*)data;
	delete [] dt;
	GL(glDeleteBuffers(1, &vbo));
}

void VertexBuffer::draw(size_t n, int type)
{
	GL(glBindBuffer(GL_ARRAY_BUFFER, vbo));
	GL(glDrawArrays(type, 0, n));
	GL(glBindBuffer(GL_ARRAY_BUFFER, vbo));
}

void VertexBuffer::subdata(void* data, size_t offset, size_t size)
{
	GL(glBindBuffer(GL_ARRAY_BUFFER, vbo));
	assert(size + offset <= this->size);
	std::memcpy((uint8_t*)this->data + offset, data, size);
	GL(glBufferSubData(GL_ARRAY_BUFFER, offset, size, data));
	GL(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void VertexBuffer::with(const std::function<void()>& fn)
{
	GL(glBindBuffer(GL_ARRAY_BUFFER, vbo));
	fn();
	GL(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

VertexArray::VertexArray()
{
	GL(glGenVertexArrays(1, &vao));
}

VertexArray::~VertexArray()
{
	GL(glDeleteVertexArrays(1, &vao));
}

void VertexArray::with(const std::function<void()>& fn)
{
	GL(glBindVertexArray(vao));
	fn();
	GL(glBindVertexArray(0));
}

void VertexArray::draw(size_t n, int type)
{
	GL(glBindVertexArray(vao));
	GL(glDrawArrays(type, 0, n));
	GL(glBindVertexArray(0));
}

VertexArray::VertexArray(const std::function<void()>& fn)
{
	GL(glGenVertexArrays(1, &vao));
	GL(glBindVertexArray(vao));
	with(fn);
	GL(glBindVertexArray(0));
}

Texture::Texture(int textureType) : textureType(textureType)
{
	GL(glGenTextures(1, &textureId));
}

Texture::~Texture()
{
	GL(glDeleteTextures(1, &textureId));
}

void Texture::with(const std::function<void()>& fn, int slot)
{
	GL(glActiveTexture(slot + GL_TEXTURE0));
	GL(glBindTexture(textureType, textureId));
	fn();
	GL(glBindTexture(textureType, 0));
}

Texture2D::Texture2D(int width, int height, GLenum internalFormat) : Texture(GL_TEXTURE_2D), width(width), height(height), internalFormat(internalFormat)
{
	with([&]()
	{
		GL(glTexStorage2D(GL_TEXTURE_2D, 1, internalFormat, width, height));
	});
}

void Texture2D::setData(void* data, GLenum dataFormat, GLenum dataType)
{
	with([&]()
	{
		GL(glTexSubImage2D(GL_TEXTURE_2D, 0,0,0, width, height, dataFormat, dataType, data));
	}, 0);
}

void Texture::setFiltering(int filter)
{
	with([&]()
	{
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter));
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter));
		this->filter = filter;
	}, 0);
}

void Texture2D::setWrapping(int wrapX, int wrapY)
{
	with([&]()
	{
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapX));
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapY));
		this->wrapX = wrapX;
		this->wrapY = wrapY;
	}, 0);
}

Texture1D::Texture1D(int width, GLenum internalFormat) : Texture(GL_TEXTURE_1D), width(width), internalFormat(internalFormat)
{
	with([&]()
	{
		GL(glTexStorage1D(GL_TEXTURE_1D, 1, internalFormat, width));
	});
}

void Texture1D::setData(void* data, GLenum dataFormat, GLenum dataType)
{
	with([&]()
	{
		GL(glTexSubImage1D(GL_TEXTURE_1D, 0, 0, width, dataFormat, dataType, data));
	}, 0);
}

void Texture1D::setWrapping(int wrapX)
{
	with([&]()
	{
		GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, wrapX));
		this->wrapX = wrapX;
	}, 0);
}

void Texture::withMultiple(const std::vector<std::shared_ptr<Texture>>& textures, const std::function<void()>& fn)
{
	int slot = 0;
	for(auto t : textures)
	{
		GL(glActiveTexture(slot + GL_TEXTURE0));
		GL(glBindTexture(t->textureType, t->textureId));
		slot++;
	}
	fn();
	for(auto t : textures)
	{
		slot--;
		GL(glActiveTexture(slot + GL_TEXTURE0));
		GL(glBindTexture(t->textureType, 0));
	}
}
