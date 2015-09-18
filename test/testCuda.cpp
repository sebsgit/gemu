#include "cuda/cudaForward.h"
#include "../ptx/runtime/PtxExecutionContext.h"
#include "../arch/Device.h"
#include <cassert>
#include <iostream>

#define cu_assert(x) (assert((x) == CUDA_SUCCESS_))

static void test_device() {
	int count = -1;
	int driverVersion = -1;
	cu_assert(cuDriverGetVersion(&driverVersion));
	assert(driverVersion > 0);
	cu_assert(cuDeviceGetCount(&count));
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

static void test_module() {
	const std::string test_source =
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry kernel(.param .u64 kernel_param_0) {\n"
	".reg .s32 	%r<2>;\n"
	".reg .s64 	%rd<3>;\n"
	"ld.param.u64 	%rd1, [kernel_param_0];\n"
	"cvta.to.global.u64 	%rd2, %rd1;\n"
	"mov.u32 	%r1, 5;\n"
	"st.global.u32 	[%rd2], %r1;\n"
	"ret;\n"
	"}\n";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, test_source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel"));
	assert(cuModuleGetFunction(&funcHandle, modId, "nosuchkernel___123") == CUDA_ERROR_NOT_FOUND_);
	cu_assert(cuModuleUnload(modId));
	assert(cuModuleGetFunction(&funcHandle, modId, "kernel") != CUDA_SUCCESS_);
}

void test_cuda(){
	std::cout << "testing cuda...\n";
	assert(cuInit(0) == CUDA_SUCCESS_);
	test_device();
	test_memory();
	test_module();
	std::cout << "done.\n";
}
