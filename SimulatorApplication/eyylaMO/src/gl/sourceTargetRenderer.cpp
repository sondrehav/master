#include "sourceTargetRenderer.h"
#include "debug.h"
#include <glad/glad.h>
#include "shader.h"
#include <glm/glm.hpp>

static const std::string vertexSource = "#version 450 core\n\nlayout(location = 2) in vec2 position; \n\nuniform mat4 matrix = mat4(1.0f); \n\nvoid main() { \n	gl_Position = matrix * vec4(position, 0.0, 1.0); \n }";
static const std::string fragmentSource = "#version 450 core\n\nlayout (location = 0) out vec4 color; \n\nuniform vec3 inputColor;\n\nvoid main(){\n\n	if(length(gl_PointCoord - vec2(0.5)) >= 0.25) discard;\n	color = vec4(inputColor, alphaValue);\n}";

SourceTargetRenderer::SourceTargetRenderer()
{
	GL(glGenBuffers(1, &vbuffer));
	GL(glBindBuffer(GL_ARRAY_BUFFER, vbuffer));

	glm::vec2 vertex(0, 0);
	GL(glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2), &vertex, GL_STATIC_DRAW));

	GL(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void*)(0)));
	GL(glEnableVertexAttribArray(0));

	GL(glBindBuffer(GL_ARRAY_BUFFER, 0));

	shader = std::make_unique<Shader>();
	GL(shader->attach(vertexSource, GL_VERTEX_SHADER));
	GL(shader->attach(fragmentSource, GL_FRAGMENT_SHADER));
	GL(shader->link());
	GL(shader->validate());


}

void SourceTargetRenderer::render(const std::vector<glm::ivec2>& sources, const std::vector<glm::ivec2>& destinations,
	const glm::mat4& matrix)
{
	shader->use();
	int matrixLocation = shader->getUniformLocation("matrix");
	int colorLocation = shader->getUniformLocation("inputColor");

	glm::vec3 color = glm::vec3(0.2, 0.2, 1.0);

	GL(glUniformMatrix4fv(matrixLocation, 1, false, (float*)&matrix));
	GL(glUniform3fv(colorLocation, 1, (float*)&color));

	GL(glPointSize(10.0f));

	GL(glBindBuffer(GL_ARRAY_BUFFER, vbuffer));
	GL(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (GLvoid*)(0)));
	GL(glEnableVertexAttribArray(0));
	GL(glDrawArrays(GL_POINTS, 0, 1));
	GL(glDisableVertexAttribArray(0));
	GL(glBindBuffer(GL_ARRAY_BUFFER, 0));
	GL(glDisable(GL_BLEND));
}
