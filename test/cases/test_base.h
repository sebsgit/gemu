#ifndef GEMUTESTCASEBASE_H
#define GEMUTESTCASEBASE_H

#include <cassert>
#include <iostream>
#include <cuda.h>
#include <cmath>

#define cu_assert(x) (assert((x) == CUDA_SUCCESS))
#define assert_float_eq(x,y) (assert(fabs(x-y) < 0.001f))

void init_test() {
	cuInit(0);
	int devId;
	CUcontext context;
	cuDeviceGet(&devId,0);
	cuCtxCreate(&context,0,devId);
}

#endif
