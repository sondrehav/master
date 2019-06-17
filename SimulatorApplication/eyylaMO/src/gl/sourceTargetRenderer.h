#pragma once
#include "shader.h"
#include "texture.h"
#include <glm/glm.hpp>
#include <vector>

class SourceTargetRenderer
{

public:
	SourceTargetRenderer();
	~SourceTargetRenderer();

	void render(const std::vector<glm::ivec2>& sources, const std::vector<glm::ivec2>& destinations, const glm::mat4& matrix = glm::mat4(1.0f));

private:
	std::unique_ptr<Shader> shader;
	GLuint vbuffer;

};
