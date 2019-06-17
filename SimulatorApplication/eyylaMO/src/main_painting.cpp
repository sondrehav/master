#include "main.h"
#include "cudaDebug.h"
#include "helper_math.h"
#include "../imgui/imgui_helper.h"


void WaveSolver::onMouseMove(double x, double y)
{
	mousePosition.x = x;
	mousePosition.y = y;
	if (brushState == Painting && paintOperation != NonePaintOperation)
	{
		printf("%s at (%f, %f)\n", paintOperation == Add ? "Adding" : "Removing", x, y);
	}
	auto& io = ImGui::GetIO();
	io.MousePos.x = x;
	io.MousePos.y = y;
}

void WaveSolver::onMouseDown(double x, double y, int mouseNum, int mods)
{
	if(brushState == NoneBrushState && mouseNum == GLFW_MOUSE_BUTTON_LEFT)
	{
		brushState = Painting;
	} else if(brushState == NoneBrushState && mouseNum == GLFW_MOUSE_BUTTON_RIGHT)
	{
		brushState = Line;
		initialLineLocation.x = x;
		initialLineLocation.y = y;
	}

	auto& io = ImGui::GetIO();
	io.MouseDown[mouseNum] = 1;
}

void WaveSolver::onMouseUp(double x, double y, int mouseNum, int mods)
{
	if(brushState == Line && mouseNum == GLFW_MOUSE_BUTTON_RIGHT)
	{
		if(paintOperation != NonePaintOperation)
		{
			printf("%s line from (%f, %f) to (%f, %f)\n", paintOperation == Add ? "Adding" : "Removing", initialLineLocation.x, initialLineLocation.y, x, y);

			// int2 from, int2 to, float brushSize, float brushFalloff, int width, int height
			float2 mult = make_float2((float)width / windowWidth, (float)height / windowHeight);

			int2 from = make_int2(initialLineLocation.x * mult.x, initialLineLocation.y * mult.y);
			int2 to = make_int2(x * mult.x, y * mult.y);

			float falloff = brushSize / 2.0f;

			void *args[4] = { (void*)&from, (void*)&to, (void*)&brushSize, (void*) &falloff};

			dim3 ddimBlock = dim3(32, 32, 1);
			dim3 ddimGrid = dim3((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
			
			CUDA_D(cuLaunchKernel(drawLineFunction, ddimGrid.x, ddimGrid.y, ddimGrid.z, ddimBlock.x, ddimBlock.y, ddimBlock.z, 0, NULL, args, NULL));
			CUDA(cudaDeviceSynchronize());
		}
		brushState = NoneBrushState;
	}
	if (brushState == Painting && mouseNum == GLFW_MOUSE_BUTTON_LEFT)
	{
		brushState = NoneBrushState;
	}
	auto& io = ImGui::GetIO();
	io.MouseDown[mouseNum] = 0;
}

void WaveSolver::onMouseScroll(double dx, double dy)
{
	brushSize = fmax(fmin(100.0f, brushSize + dy), 1.0f);
}

void WaveSolver::onMouseExit()
{
	printf("MOUSE OUT\n");
	brushState = NoneBrushState;
}