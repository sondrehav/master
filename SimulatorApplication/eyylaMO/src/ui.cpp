#include "main.h"

#include "../imgui/imgui.h"
#include "../imgui/imgui_helper.h"
#include "../imgui/imgui_impl_gl.h"
#include "misc/winDialogs.h"
#include "../audio/AudioFile.h"
#include "cudaDebug.h"
#include "../imgui/imgui_internal.h"

void WaveSolver::beginUI()
{
	ImGuiH::Init(windowWidth, windowHeight, this);
	ImGui::InitGL();
}

void WaveSolver::renderUI(int width, int height, double dt)
{

	// Update imgui configuration
	auto &imgui_io = ImGui::GetIO();
	imgui_io.DeltaTime = static_cast<float>(dt);
	imgui_io.DisplaySize = ImVec2(width, height);

	ImGui::NewFrame();
	ImGui::SetNextWindowBgAlpha(0.5);
	ImGui::SetNextWindowSize(ImVec2(240, 0), ImGuiCond_FirstUseEver);

	if (ImGui::Begin("WaveSolver", nullptr))
	{

		if (ImGui::Button("Load"))
		{
			openDialog([&](bool aborted, const std::string& filepath)
			{
				if (!aborted)
				{
					AudioFile<float>* file = new AudioFile<float>;
					if (file->load(filepath))
					{
						if (inputFile != nullptr) delete inputFile;
						inputFile = file;
						file->printSummary();
						initializeSimulation();
					}
					else delete file;
				}
			});
		}

		if (inputFile == nullptr)
		{
			ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
		}

		if (iteration == 0 || simulate)
		{
			ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
		}
		if (ImGui::Button("Save"))
		{
			saveDialog([&](bool aborted, const std::string& filepath)
			{
				if (!aborted)
				{
					int numSamples = iteration / stepsPerSample;
					float* data = new float[numSamples * numOutputChannels];

					float* tempData = new float[numOutputChannels * numOutputSamples];
					CUDA_D(cuMemcpyDtoH_v2(tempData, waveOutput, numOutputSamples * sizeof(float) * numOutputChannels));

					for(int i = 0; i < numOutputChannels; i++)
					{
						std::memcpy(data + i * numSamples, tempData + i * numOutputSamples, numSamples*sizeof(float));
					}

					std::vector<std::vector<float>> hostData;
					for(int channel = 0; channel < numOutputChannels; channel++)
					{
						std::vector<float> samples(numSamples);
						std::memcpy(samples.data(), data + numSamples * channel, numSamples * sizeof(float));
						hostData.push_back(samples);
					}
					delete[] data;
					delete[] tempData;
					
					AudioFile<float> f;
					AudioFile<float>::AudioBuffer buffer(hostData);
					f.setSampleRate(sampleRate);
					f.setBitDepth(inputFile->getBitDepth());
					f.setAudioBufferSize(numOutputChannels, numSamples);
					if(!f.setAudioBuffer(buffer))
					{
						errorMsg("Could not set audio buffer!");
						return;
					}
					f.printSummary();

					printf("%d, %d, %d, %d, \n", (int)numSamples, (int)numOutputChannels, (int)buffer.size(), (int)buffer[0].size());

					if (!f.save(filepath))
					{
						errorMsg("Could not save file!");
					}

				}
			});
		}

		if (iteration == 0 || simulate)
		{
			ImGui::PopItemFlag();
		}


		ImGui::Checkbox("Simulating", &simulate);
		ImGui::LabelText("#sample", "%d of %d", iteration / stepsPerSample, (int)numOutputSamples);
		ImGui::LabelText("Hz", "%5.1f", currentFrequency);
		if (ImGui::Button("Reset"))
		{
			resetSimulation();
		}
		if (ImGui::Button("Clear geometry"))
		{
			clearGeometry();
		}
		if (inputFile == nullptr)
		{
			ImGui::PopItemFlag();
		}


		ImGui::Separator();

		ImGui::SliderFloat("Thickness", &brushSize, 1.0, 20.0);
		ImGui::SliderFloat("Wall absorbtion", &wallAbsorbtion, 0.0, 1.0);
		ImGui::SliderFloat("Tail (seconds)", &tail, 0.0, 20.0);
		ImGui::SliderFloat("Amp", &amp, 1.0, 15);
		ImGui::SliderInt("StepsPerSample", &stepsPerSample, 1, 32);
	}
	ImGui::End();

	ImGui::Render();
	ImDrawData *imguiDrawData;
	imguiDrawData = ImGui::GetDrawData();
	ImGui::RenderDrawDataGL(imguiDrawData);
	ImGui::EndFrame();
}

