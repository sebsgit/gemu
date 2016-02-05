#ifndef GEMUTESTCASEBASE_H
#define GEMUTESTCASEBASE_H

#include <cassert>
#include <iostream>
#include <cuda.h>
#include <cmath>

#define cu_assert(x) {auto t = (x); if (t != CUDA_SUCCESS)std::cout << t << '\n'; (assert((t) == CUDA_SUCCESS));}
#define assert_float_eq(x,y) (assert(fabs(x-y) < 0.001f))

void init_test() {
	cu_assert(cuInit(0));
	int devId;
	CUcontext context;
	cu_assert(cuDeviceGet(&devId,0));
	cu_assert(cuCtxCreate(&context,0,devId));
}

#endif
