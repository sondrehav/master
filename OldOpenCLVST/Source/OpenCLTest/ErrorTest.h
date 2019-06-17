#pragma once

#include <iostream>

#define CL_CHECK_ERR(x) if (err != CL_SUCCESS) { std::cout << "Error code: " << err << std::endl; __debugbreak(); abort(); }

#ifdef _DEBUG
#define CL(x) {\
	cl_int err = x;\
	CL_CHECK_ERR(err)\
	}
#else
#define CL(x) {\
	cl_int err = x;\
	}
#endif

#define assert(x) if(!x) { abort(); }