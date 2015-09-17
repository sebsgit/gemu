#include "cuda/cudaForward.h"
#include <cassert>
#include <iostream>

static void test_device() {
	int count = -1;
	int driverVersion = -1;
	assert(cuDriverGetVersion(&driverVersion) == CUDA_SUCCESS_);
	assert(driverVersion > 0);
	assert(cuDeviceGetCount(&count) == CUDA_SUCCESS_);
	assert(count > 0);
	CUdevice devId = 0;
	assert(cuDeviceGet(&devId, 0) == CUDA_SUCCESS_);
	char name[512];
	assert(cuDeviceGetName(name,512,devId) == CUDA_SUCCESS_);
	size_t totalMemory=0;
	assert(cuDeviceTotalMem(&totalMemory,devId)==CUDA_SUCCESS_);
	assert(totalMemory > 0);
	std::cout << "using device: " << name << " with " << (totalMemory/(1024.0f * 1024)) << " mb memory\n";
}

static void test_memory(){
	int to_test = 5, to_get = -1;
	CUdeviceptr dev_value;
	assert(cuMemAlloc(&dev_value,sizeof(to_test)) == CUDA_SUCCESS_);
	assert(cuMemcpyHtoD(dev_value, &to_test, sizeof(to_test)) == CUDA_SUCCESS_);
	assert(cuMemcpyDtoH(&to_get, dev_value, sizeof(to_get)) == CUDA_SUCCESS_);
	assert(to_get == to_test);
	assert(cuMemFree(dev_value) == CUDA_SUCCESS_);
}

void test_cuda(){
	std::cout << "testing cuda...\n";
	assert(cuInit(0) == CUDA_SUCCESS_);
	test_device();
	test_memory();
	std::cout << "done.\n";
}
