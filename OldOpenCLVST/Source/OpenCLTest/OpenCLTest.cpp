#include "OpenCLTest.h"
#include <functional>
#include "ErrorTest.h"

compute::CLContext::CLContext(const std::string& platform, const std::string& device) 
{
	setup(platform, device);
}

compute::CLContext::~CLContext()
{
	destroy();
}


void compute::CLContext::setup(const std::string& platform, const std::string& device)
{
	assert(selectPlatformByName(platform, &mPlatform));
	assert(selectDeviceByName(device, mPlatform, &mDevice));

	cl_int err;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)mPlatform, 0 };

	mContext = clCreateContext(props, 1, &mDevice, NULL, NULL, &err);
	CL_CHECK_ERR(err);

	mQueue = clCreateCommandQueue(mContext, mDevice, 0, &err);
	CL_CHECK_ERR(err);
}

void compute::CLContext::destroy()
{
	CL(clFlush(mQueue));
	CL(clFinish(mQueue));
	CL(clReleaseCommandQueue(mQueue));
	CL(clReleaseContext(mContext));
}

char* compute::CLContext::getPlatformString(cl_platform_id platform, cl_platform_info info)
{
	size_t str_len;
	CL(clGetPlatformInfo(platform, info, NULL, NULL, &str_len));
	char* name = new char[str_len];
	CL(clGetPlatformInfo(platform, info, str_len * sizeof(char), name, NULL));
	return name;
}

char* compute::CLContext::getDeviceString(cl_device_id device, cl_device_info info)
{
	size_t str_len;
	CL(clGetDeviceInfo(device, info, NULL, NULL, &str_len));
	char* name = new char[str_len];
	CL(clGetDeviceInfo(device, info, str_len * sizeof(char), name, NULL));
	return name;
}

bool compute::CLContext::selectPlatformByName(const std::string& platformName, cl_platform_id* platform)
{

	cl_uint ret_num_platforms;
	CL(clGetPlatformIDs(0, NULL, &ret_num_platforms));

	cl_platform_id* platform_ids = new cl_platform_id[ret_num_platforms];
	CL(clGetPlatformIDs(ret_num_platforms, platform_ids, NULL));

	for (int i = 0; i < ret_num_platforms; i++)
	{
		cl_platform_id pid = platform_ids[i];
		char* name = getPlatformString(platform_ids[i], CL_PLATFORM_NAME);
		if (std::string(name) == platformName)
		{
			delete[] name;
			delete[] platform_ids;
			*platform = pid;
			return true;
		}
		delete[] name;
	}
	delete[] platform_ids;
	return false;

}

bool compute::CLContext::selectDeviceByName(const std::string& deviceName, cl_platform_id platform, cl_device_id* id)
{
	cl_uint ret_num_devices;
	CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &ret_num_devices));

	cl_device_id* device_ids = new cl_device_id[ret_num_devices];
	CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, ret_num_devices, device_ids, NULL));

	for (int i = 0; i < ret_num_devices; i++)
	{
		cl_device_id did = device_ids[i];
		char* name = getDeviceString(device_ids[i], CL_DEVICE_NAME);
		if (std::string(name) == deviceName)
		{
			delete[] name;
			delete[] device_ids;
			*id = did;
			return true;
		}
		delete[] name;
	}
	delete[] device_ids;
	return false;
}
