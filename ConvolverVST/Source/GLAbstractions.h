#pragma once

#include <functional>
#include <vector>
#include <memory>

class VertexBuffer
{
public:
	VertexBuffer(size_t size);

	VertexBuffer(const VertexBuffer& other) = delete;
	VertexBuffer& operator=(const VertexBuffer&) = delete; // non copyable

	~VertexBuffer();
	void draw(size_t n, int type);
	void subdata(void* data, size_t offset, size_t size);
	void with(const std::function<void()>& fn);
private:
	unsigned int vbo;
	void* data;
	const size_t size;
};

class VertexArray
{
public:
	VertexArray();
	VertexArray(const std::function<void()>& fn);

	VertexArray(const VertexBuffer& other) = delete;
	VertexArray& operator=(const VertexArray&) = delete; // non copyable

	~VertexArray();
	void with(const std::function<void()>& fn);
	void draw(size_t n, int type);
private:
	unsigned int vao;
};


class Texture
{
public:
	Texture(int textureType);
	virtual ~Texture();

	Texture(const Texture& other) = delete;
	Texture& operator=(const Texture&) = delete; // non copyable

	void with(const std::function<void()>& fn, int slot = 0);

	virtual void setData(void* data, GLenum dataFormat, GLenum dataType) = 0;

	void setFiltering(int filter);

	static void withMultiple(const std::vector<std::shared_ptr<Texture>>&, const std::function<void()>&);

private:
	const int textureType;
	unsigned int textureId = 0;
	int filter;

};


class Texture2D : public Texture
{
public:
	Texture2D(int width, int height, GLenum internalFormat = GL_RGBA8);

	void setData(void* data, GLenum dataFormat, GLenum dataType) override;
	void setWrapping(int wrapX, int wrapY);

	const int width;
	const int height;

private:
	const int internalFormat;
	int wrapX;
	int wrapY;

};

class Texture1D : public Texture
{
public:
	Texture1D(int width, GLenum internalFormat = GL_RGBA8);

	void setData(void* data, GLenum dataFormat, GLenum dataType) override;
	void setWrapping(int wrapX);

	const int width;

private:
	const int internalFormat;
	int wrapX;

};