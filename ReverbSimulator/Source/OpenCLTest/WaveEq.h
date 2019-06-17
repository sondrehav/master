#pragma once

#define CLBLAST(x) {\
	clblast::StatusCode err = x;\
	if(err != clblast::StatusCode::kSuccess) { std::cout << "Error code: " << (int)err << std::endl; __debugbreak(); abort(); }\
	}
#include "Solver.h"
#include <CL/cl.h>
#include <memory>

namespace compute {

	class CLContext;

class PUBLIC_API Solver
{
public: 
	Solver(int dimension, CLContext& ctx);
		
	~Solver();

	/*
	 * Un = pow(r,2) * (U.dot(D) + D.dot(U)) + 2 * U - Ul
	 */
	void step();

	void outputResults();

	std::shared_ptr<float[]> getContents(size_t* size);

	int getWidth() { return dimension; }
	int getHeight() { return dimension; }
	

private:
	std::shared_ptr<cl_float[]> m_U;
	cl_float* m_Diag;

	cl_mem bufs_U[3] = {0, 0, 0};

	cl_mem buf_D;

	CLContext& ctx;

	const int dimension;

	float c = 1;				// stiffness
	float k = 256.0/44100.0;				// timestep
	float h;					// spacial step
	float floorSizeTotal = 64;

	int iteration = 0;
	cl_event previous_cycle = NULL;

};


}
