#include "editorRenderer.h"
#include "debug.h"
#include <glad/glad.h>
#include "shader.h"
#include <glm/gtc/matrix_transform.hpp>

static const std::string areaVertSource = "#version 450 core\n\nlayout (location = 0) in vec2 position;\n\nuniform mat4 matrix;\nout vec2 pass_position;\n\nvoid main()\n{\n	pass_position = position;\n	gl_Position = matrix * vec4(position, 0.0, 1.0);\n}";
static const std::string areaFragSource = "#version 450 core\n\nlayout (location = 0) out vec4 color; \n\nuniform vec3 inputColor;\n\nin vec2 pass_position;\n\nvoid main(){\n	vec2 value = pass_position - vec2(0.5);\n    float alphaValue = min(2.0f * max(1.0f - 2.0 * length(value), 0.0f), 1.0f);\n	color = vec4(inputColor, alphaValue);\n}\n";

static const std::string lineVertSource = "#version 450 core\n\nlayout (location = 0) in vec2 position;\n\nuniform mat4 matrix;\n\nvoid main()\n{\n	gl_Position = matrix * vec4(position, 0.0, 1.0);\n}";
static const std::string lineFragSource = "#version 450 core\n\nlayout (location = 0) out vec4 colorOut;\n\n\n\nvoid main()\n{\ncolorOut = vec4(1.0);\n}";

EditorRenderer::EditorRenderer()
{
	GL(glGenBuffers(1, &lineVertexBufferId));
	GL(glBindBuffer(GL_ARRAY_BUFFER, lineVertexBufferId));
	GL(glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * 2, line, GL_DYNAMIC_DRAW));

	GL(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void*)(0)));
	GL(glEnableVertexAttribArray(0));

	GL(glBindBuffer(GL_ARRAY_BUFFER, 0));

	areaShader = std::make_unique<Shader>();
	GL(areaShader->attach(areaVertSource, GL_VERTEX_SHADER));
	GL(areaShader->attach(areaFragSource, GL_FRAGMENT_SHADER));
	GL(areaShader->link());
	GL(areaShader->validate());

	lineShader = std::make_unique<Shader>();
	GL(lineShader->attach(lineVertSource, GL_VERTEX_SHADER));
	GL(lineShader->attach(lineFragSource, GL_FRAGMENT_SHADER));
	GL(lineShader->link());
	GL(lineShader->use());
	GL(lineShader->validate());

	GL(glGenBuffers(1, &vertexBufferId));
	GL(glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId));
	GL(glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * 6, (GLvoid*)&vertexData, GL_STATIC_DRAW));

	GL(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void*)(0)));
	GL(glEnableVertexAttribArray(0));

	GL(glBindBuffer(GL_ARRAY_BUFFER, 0));

}

void EditorRenderer::render(BrushState brushState, PaintOperation paintOperation, glm::vec2 initialLineLocation, glm::vec2 mousePosition, float brushSize, int width, int height)
{

	GL(glDisable(GL_DEPTH_TEST));
	if(brushState == Line)
	{
		glm::mat4 matrix = glm::scale(glm::mat4(1.0), glm::vec3(2.0 / width, -2.0 / height, 1.0f));
		matrix = glm::translate(matrix, glm::vec3(-width / 2, -height / 2, 0.0));

		GL(glLineWidth(3.0f));
		lineShader->use();
		int matrixLocation = lineShader->getUniformLocation("matrix");
		GL(glUniformMatrix4fv(matrixLocation, 1, false, (GLfloat*) &matrix));

		line[0] = initialLineLocation;
		line[1] = mousePosition;
		GL(glBindBuffer(GL_ARRAY_BUFFER, lineVertexBufferId));
		GL(glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(glm::vec2) * 2, line));

		GL(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (GLvoid*)(0)));
		GL(glEnableVertexAttribArray(0));

		GL(glDrawArrays(GL_LINES, 0, 4));

		GL(glDisableVertexAttribArray(0));
		GL(glBindBuffer(GL_ARRAY_BUFFER, 0));
	}

	if(paintOperation != NonePaintOperation)
	{
		glm::mat4 modelMatrix = glm::mat4(1.0);
		modelMatrix = glm::translate(modelMatrix, glm::vec3(mousePosition.x, mousePosition.y, 0));
		modelMatrix = glm::scale(modelMatrix, glm::vec3(brushSize));
		modelMatrix = glm::translate(modelMatrix, glm::vec3(-0.5, -0.5, 0));
		
		glm::mat4 projectionMatrix = glm::mat4(1.0f);
		projectionMatrix = glm::scale(projectionMatrix, glm::vec3(2.0 / width, -2.0 / height, 0.0));
		projectionMatrix = glm::translate(projectionMatrix, glm::vec3(-width / 2, -height / 2, 0.0));
		//projectionMatrix = glm::translate(projectionMatrix, glm::vec3(-1, -1, 0.0));

		glm::mat4 matrix = projectionMatrix * modelMatrix;

		GL(glEnable(GL_BLEND));
		GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

		areaShader->use();
		int matrixLocation = areaShader->getUniformLocation("matrix");
		int colorLocation = areaShader->getUniformLocation("inputColor");

		glm::vec3 color = glm::vec3(paintOperation == Add ? 0.0 : 1.0, paintOperation == Add ? 1.0 : 0.0, 0.0);
		
		GL(glUniformMatrix4fv(matrixLocation, 1, false, (float*)&matrix));
		GL(glUniform3fv(colorLocation, 1, (float*)&color));

		GL(glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId));
		GL(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (GLvoid*)(0)));
		GL(glEnableVertexAttribArray(0));
		GL(glDrawArrays(GL_TRIANGLES, 0, 6));
		GL(glDisableVertexAttribArray(0));
		GL(glBindBuffer(GL_ARRAY_BUFFER, 0));
		GL(glDisable(GL_BLEND));
	}

	

}

EditorRenderer::~EditorRenderer()
{
	areaShader.reset();
	lineShader.reset();
	GL(glDeleteBuffers(1, &vertexBufferId));
	GL(glDeleteBuffers(1, &lineVertexBufferId));
}
