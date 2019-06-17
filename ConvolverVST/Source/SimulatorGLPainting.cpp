#include <glad/glad.h>
#include "SimulatorGL.h"
#include "debug.h"

void SimulatorGLComponent::mouseDrag(const MouseEvent& event)
{
	paintBrushLocation = event.position;
	if(cursorDown)
	{
		printf("painting at %d, %d\n", event.x, event.y);
		Point<float> p;
		p.x = width * paintBrushLocation.x / getWidth();
		p.y = height * paintBrushLocation.y / getHeight();
		float amt = paintAmount;
		if (event.mods.isRightButtonDown()) amt = -amt;
		processor.getSimulator().drawGeometryAt(p, paintSize, amt, paintFalloff);
	}
	context.triggerRepaint();
}

void SimulatorGLComponent::mouseDown(const MouseEvent& event)
{
	cursorDown = true;
	printf("cursor down\n");
	context.triggerRepaint();
}

void SimulatorGLComponent::mouseUp(const MouseEvent& event)
{
	cursorDown = false;
	printf("cursor up\n");
	context.triggerRepaint();
}

void SimulatorGLComponent::mouseWheelMove(const MouseEvent& event, const MouseWheelDetails& wheel)
{
	if(event.mods.isCtrlDown())
	{
		paintAmount = std::clamp<float>(paintAmount + wheel.deltaY * 0.1, 0, 1);
		printf("paint amount: %f\n", paintAmount);
	} else if(event.mods.isShiftDown())
	{
		paintFalloff = std::clamp<float>(paintFalloff + wheel.deltaY * 0.1, 0, 1);
		printf("paint falloff: %f\n", paintFalloff);
	} else
	{
		paintSize = std::clamp<float>(paintSize + wheel.deltaY, 1, 100);
		printf("paint size: %f\n", paintSize);
	}
	context.triggerRepaint();
}

void SimulatorGLComponent::mouseMove(const MouseEvent& event)
{
	paintBrushLocation = event.position;
	context.triggerRepaint();
}

void SimulatorGLComponent::renderGeometry(const float* matrix)
{
	float mm[16];
	std::memcpy(mm, matrix, sizeof(float) * 16);
	screenShader->with([&]()
	{
		int location = screenShader->getUniformLocation("transform");
		GL(glUniformMatrix4fv(location, 1, false, (GLfloat*)&mm));
		GL(glUniform1i(screenShader->getUniformLocation("mode"), 0));
		GL(glUniform1f(screenShader->getUniformLocation("z"), -0.1f));
		Texture::withMultiple({ geometryTexture, geometryLutTexture }, [&]()
		{
			screenQuadArray->draw(6, GL_TRIANGLES);
		});
	});
}

void SimulatorGLComponent::renderPaintBrush(const float* matrix)
{
	float mm[16] = {
		4.0 / width, 0.0, 0.0, -1.0,
		0.0, -4.0 / height, 0.0, 1.0,
		0.0, 0.0, 1.0, 0.0, 
		0.0, 0.0, 0.0, 1.0,
	};
	float pos[2];
	pos[0] = width * paintBrushLocation.x / getWidth();
	pos[1] = height * (paintBrushLocation.y / getHeight());
	printf("%f, %f\n", pos[0], pos[1]);
	paintingBuffer->subdata(pos, 0, 2 * sizeof(float));
	paintingShader->with([&]()
	{
		GL(glUniform1f(paintingShader->getUniformLocation("amount"), paintAmount));
		GL(glUniform1f(paintingShader->getUniformLocation("falloff"), paintFalloff));
		GL(glUniform1f(paintingShader->getUniformLocation("size"), paintSize));
		GL(glUniformMatrix4fv(paintingShader->getUniformLocation("transform"), 1, true, (GLfloat*)&mm));
		paintingArray->draw(1, GL_POINTS);
	});
}