#include "Convolve.h"

cl_int compute::convolve(cl_context ctx, cl_command_queue queue , const size_t width, float* src, float* dst, size_t convolutionSize, float* convolution)
{
	assert(convolutionSize % 2 == 1);

	cl_int err;

	std::ifstream kernelIn("convolve1d.ocl");
	std::string kernelSource((std::istreambuf_iterator<char>(kernelIn)), std::istreambuf_iterator<char>());

	// Create the compute program from the source buffer
	cl_program program = clCreateProgramWithSource(ctx, 1,
		(const char **)kernelSource.c_str(), NULL, &err);

	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	cl_kernel kernel = clCreateKernel(program, "convolve", &err);


	// Device input buffers
	cl_mem d_src = clCreateBuffer(ctx, CL_MEM_READ_ONLY, width * sizeof(float), NULL, &err);
	cl_mem d_conv = clCreateBuffer(ctx, CL_MEM_READ_ONLY, convolutionSize * sizeof(float), NULL, &err);

	// Device output buffer
	cl_mem d_dest = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, width * sizeof(float), NULL, &err);

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, d_src, CL_TRUE, 0, width * sizeof(float), src, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_conv, CL_TRUE, 0, convolutionSize * sizeof(float), convolution, 0, NULL, NULL);

	return err;

}