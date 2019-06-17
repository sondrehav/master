#pragma once

#include <CL/cl.h>
#include <clblast.h>
#include <cassert>
#include <fstream>

namespace compute
{

cl_int convolve(cl_context ctx, cl_command_queue queue, const size_t width, float* src, float* dst, size_t convolutionSize, float* convolution);

}
