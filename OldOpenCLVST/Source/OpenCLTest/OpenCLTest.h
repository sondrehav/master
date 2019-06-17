#pragma once

#include <CL/cl.h>
#include <string>
#include "Solver.h"

// NVIDIA CUDA
// Quadro P2000 with Max-Q Design

namespace compute
{
	
class PUBLIC_API CLContext
{

public:	
	CLContext(const std::string& platform, const std::string& device);
	~CLContext();

	cl_command_queue& getQueue() { return mQueue; }
	cl_context& getContext() { return mContext; }

private:

	cl_platform_id mPlatform = NULL;
	cl_device_id mDevice = NULL;
	cl_command_queue mQueue = NULL;
	cl_context mContext = NULL;

	void setup(const std::string& platform, const std::string& device);

	void destroy();

	char* getPlatformString(cl_platform_id platform, cl_platform_info info);
	
	char* getDeviceString(cl_device_id device, cl_device_info info);

	bool selectPlatformByName(const std::string &platformName, cl_platform_id* platform);

	bool selectDeviceByName(const std::string& deviceName, cl_platform_id platform, cl_device_id* id);
	

};


}


/*
int main()
{
	


	const int size = 256;

	Solver* solver = new Solver(size, context, commandQueue);

	std::cout << "start" << std::endl;
	for(int i = 0; i < 44100; i++)
	{
		solver->step();
	}
	solver->outputResults();

	
}
*/