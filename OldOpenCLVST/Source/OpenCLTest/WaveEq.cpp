#include "WaveEq.h"
#include "OpenCLTest.h"
#include <algorithm>
#include <clblast.h>
#include "ErrorTest.h"
#include <thread>

#include <cstdlib>
#include <ctime>

compute::Solver::Solver(int dimension, compute::CLContext& ctx)
	: ctx(ctx), dimension(dimension)
{
	cl_float* ptr = new cl_float[dimension * dimension];
	m_U = std::shared_ptr<cl_float[]>(ptr);
	m_Diag = new cl_float[dimension * dimension];
	srand(time(0));

	for (int i = 0; i < dimension * dimension; i++)
	{
		int x = i % dimension - dimension / 2 - 30;
		int y = i / dimension - dimension / 2 + 94;
		m_U[i] = 1.0;
		m_Diag[i] = 0.0;
	}

	for (int i = 0; i < dimension; i++)
	{
		int index = i + dimension * i;
		if (index > 0) m_Diag[index - 1] = 1;
		m_Diag[index] = -2;
		if (index < dimension * dimension - 1) m_Diag[index + 1] = 1;
	}

	const int size = dimension * dimension;

	cl_int err;
	bufs_U[0] = clCreateBuffer(ctx.getContext(), CL_MEM_READ_WRITE, size * sizeof(float), NULL, &err);
	CL_CHECK_ERR(err);
	bufs_U[1] = clCreateBuffer(ctx.getContext(), CL_MEM_READ_WRITE, size * sizeof(float), NULL, &err);
	CL_CHECK_ERR(err);
	bufs_U[2] = clCreateBuffer(ctx.getContext(), CL_MEM_READ_WRITE, size * sizeof(float), NULL, &err);
	CL_CHECK_ERR(err);
	buf_D = clCreateBuffer(ctx.getContext(), CL_MEM_READ_ONLY, size * sizeof(float), NULL, &err);
	CL_CHECK_ERR(err);

	CL(clEnqueueWriteBuffer(ctx.getQueue(), bufs_U[0], CL_TRUE, 0, size * sizeof(float), m_U.get(), 0, NULL, NULL));
	CL(clEnqueueWriteBuffer(ctx.getQueue(), bufs_U[1], CL_TRUE, 0, size * sizeof(float), m_U.get(), 0, NULL, NULL));
	CL(clEnqueueWriteBuffer(ctx.getQueue(), bufs_U[2], CL_TRUE, 0, size * sizeof(float), m_U.get(), 0, NULL, NULL));
	CL(clEnqueueWriteBuffer(ctx.getQueue(), buf_D, CL_TRUE, 0, size * sizeof(float), m_Diag, 0, NULL, &previous_cycle));
	clFinish(ctx.getQueue());

	h = floorSizeTotal / (dimension - 1);
}

compute::Solver::~Solver()
{
	CL(clReleaseMemObject(bufs_U[0]));
	CL(clReleaseMemObject(bufs_U[1]));
	CL(clReleaseMemObject(bufs_U[2]));
	CL(clReleaseMemObject(buf_D));
	delete[] m_Diag;
}

void compute::Solver::step()
{

	cl_mem newBuf = bufs_U[(iteration + 2) % 3];
	cl_mem currentBuf = bufs_U[(iteration + 1) % 3];
	cl_mem oldBuf = bufs_U[(iteration + 0) % 3];

	// r = 0.3
	float r_sq = pow(c * k / h, 2);
	cl_event e_1;
	cl_event e_2;
	cl_event e_3;
	cl_event e_f;

	// c := alpha*a*b + beta*c for side = 'L'or'l' 
	// c := alpha*b*a + beta*c for side = 'R' or 'r'

	// need to flip the old buffer

	CL(clWaitForEvents(1, &previous_cycle));
	CL(clReleaseEvent(previous_cycle));
	CLBLAST(clblast::Axpy<float>(dimension * dimension, -2, oldBuf, 0, 1, oldBuf, 0, 1, &ctx.getQueue(), &e_f));
	CLBLAST(clblast::Symm<float>(clblast::Layout::kRowMajor, clblast::Side::kLeft, clblast::Triangle::kLower, dimension, dimension, 1, buf_D, 0, dimension, currentBuf, 0, dimension, 0, newBuf, 0, dimension, &ctx.getQueue(), &e_1));
	CLBLAST(clblast::Symm<float>(clblast::Layout::kRowMajor, clblast::Side::kRight, clblast::Triangle::kLower, dimension, dimension, r_sq, buf_D, 0, dimension, currentBuf, 0, dimension, r_sq, newBuf, 0, dimension, &ctx.getQueue(), &e_2));

	/*
	CL(clblasSsymm(clblasOrder::clblasRowMajor, clblasSide::clblasLeft, clblasUplo::clblasLower, dimension, dimension, 1, buf_D, 0, dimension, currentBuf, 0, dimension, 0, newBuf, 0, dimension, 1, &queue, 1, &previous_cycle, &event));
	CL(clblasSsymm(clblasOrder::clblasRowMajor, clblasSide::clblasRight, clblasUplo::clblasLower, dimension, dimension, r_sq, buf_D, 0, dimension, currentBuf, 0, dimension, r_sq, newBuf, 0, dimension, 1, &queue, 1, &event, &event));
	*/

	// y := y+alpha*x
	CLBLAST(clblast::Axpy<float>(dimension * dimension, 2, currentBuf, 0, 1, newBuf, 0, 1, &ctx.getQueue(), &e_3));
	//CL(clblasSaxpy(dimension * dimension, 2, currentBuf, 0, 1, newBuf, 0, 1, 1, &queue, 1, &event, &event));

	// last sub
	CL(clWaitForEvents(1, &e_3));
	CL(clWaitForEvents(1, &e_f));
	CLBLAST(clblast::Axpy<float>(dimension * dimension, 1, oldBuf, 0, 1, newBuf, 0, 1, &ctx.getQueue(), &previous_cycle));


	CL(clReleaseEvent(e_1));
	CL(clReleaseEvent(e_2));
	CL(clReleaseEvent(e_3));
	CL(clReleaseEvent(e_f));
	// at this point the new U is in newBuf...

	iteration++;

}

void compute::Solver::outputResults()
{
	cl_mem currentBuf = bufs_U[(iteration + 1) % 3];
	CL(clEnqueueReadBuffer(ctx.getQueue(), currentBuf, CL_TRUE, 0, dimension * dimension * sizeof(float), m_U.get(), 1, &previous_cycle, NULL));
	CL(clFinish(ctx.getQueue()));

	for (int i = 0; i < std::min<float>(dimension, 10); i++)
	{
		for (int j = 0; j < std::min<float>(dimension, 10); j++)
		{
			cl_float value = m_U[i*dimension + j];
			std::cout << value << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

std::shared_ptr<float[]> compute::Solver::getContents(size_t* size)
{

	std::thread::id this_id = std::this_thread::get_id();
	std::cout << __FILE__ << "(" << __LINE__ << "): thread " << this_id << std::endl;

	cl_mem currentBuf = bufs_U[(iteration + 1) % 3];
	cl_event read;
	CL(clEnqueueReadBuffer(ctx.getQueue(), currentBuf, CL_TRUE, 0, dimension * dimension * sizeof(float), m_U.get(), 1, &previous_cycle, &read));
	CL(clWaitForEvents(1, &read));
	CL(clReleaseEvent(read));

	*size = dimension * dimension;
	return m_U;
	
}

