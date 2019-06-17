#pragma once
#include "shader.h"
#include "texture.h"
#include <glm/glm.hpp>

class TextureRenderer
{

public:
	TextureRenderer();
	void render(const Texture*, const glm::mat4& matrix = glm::mat4(1.0f));

private:
	std::unique_ptr<Shader> shader;

	const float vertexData[5 * 6] = {
		 0.5f,  0.5f, 0.0f, 1.0f, 0.0f,  // top right
		 0.5f, -0.5f, 0.0f, 1.0f, 1.0f,  // bottom right
		-0.5f,  0.5f, 0.0f, 0.0f, 0.0f,  // top left 
		-0.5f, -0.5f, 0.0f, 0.0f, 1.0f,  // bottom left
		-0.5f,  0.5f, 0.0f, 0.0f, 0.0f,  // top left 
		 0.5f, -0.5f, 0.0f, 1.0f, 1.0f  // bottom right
	};

	const uint8_t lut[5*4] = {
		0x00,0x00,0x00,0xff,
		0x76,0x20,0x8c,0xff,
		0xf2,0x37,0x6c,0xff,
		0xe8,0x88,0x35,0xff,
		0xff,0xfa,0xd6,0xff
	};

	GLuint vertexBufferId;
	GLuint lutTextureId;

};
