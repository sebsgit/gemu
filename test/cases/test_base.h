#ifndef GEMUTESTCASEBASE_H
#define GEMUTESTCASEBASE_H

#include <cassert>
#include <iostream>
#include <cuda.h>

#define cu_assert(x) (assert((x) == CUDA_SUCCESS))

void init_test() {
	cuInit(0);
	int devId;
	CUcontext context;
	cuDeviceGet(&devId,0);
	cuCtxCreate(&context,0,devId);
}

#endif
