#pragma once
#include "shader.h"
#include "texture.h"
#include <glm/glm.hpp>

enum PaintOperation
{
	NonePaintOperation, Add, Subtract
};

enum BrushState
{
	NoneBrushState, Painting, Line
};

class EditorRenderer
{

public:
	EditorRenderer();
	~EditorRenderer();

	void render(BrushState brushState, PaintOperation paintOperation, glm::vec2 initialLineLocation, glm::vec2 mousePosition, float brushSize, int width, int height);

private:
	std::unique_ptr<Shader> lineShader;
	std::unique_ptr<Shader> areaShader;

	GLuint vertexBufferId;
	GLuint lineVertexBufferId;

	glm::vec2 line[2];
	glm::vec2 vertexData[6]
	{
		glm::vec2(1.0f, 1.0f),
		glm::vec2(1.0f, 0.0f),
		glm::vec2(0.0f, 1.0f),
		glm::vec2(0.0f, 0.0f),
		glm::vec2(0.0f, 1.0f),
		glm::vec2(1.0f, 0.0f)
	};

};
