#include "cuda/cudaForward.h"
#include "../arch/Device.h"
#include "cuda/cudaThreads.h"
#include <cassert>
#include <iostream>

#define cu_assert(x) (assert((x) == CUDA_SUCCESS_))

static void test_grid(){
	using namespace gemu::cuda;
	const dim3 gridSize(10, 10, 2);
	const dim3 blockSize(32, 32, 5);
	ThreadGrid grid(gridSize, blockSize);
	assert(grid.blockCount() == gridSize.x * gridSize.y * gridSize.z);
	assert(grid.block(0,0,0));
	assert(grid.block(0,0,0)->size() == blockSize);
	assert(grid.block(gridSize.x + 1, 0, 0) == nullptr);
	for (size_t gx = 0 ; gx < gridSize.x ; ++gx) {
		for (size_t gy = 0 ; gy < gridSize.y ; ++gy) {
			for (size_t gz = 0 ; gz < gridSize.z ; ++gz) {
				auto block = grid.block(gx, gy, gz);
				assert(block);
				assert(block->size() == blockSize);
				assert(block->pos() == dim3(gx, gy, gz));
				assert(block->threadCount() == blockSize.x * blockSize.y * blockSize.z);
				for (size_t bx = 0 ; bx < blockSize.x ; ++bx) {
					for (size_t by = 0 ; by < blockSize.y ; ++by) {
						for (size_t bz = 0 ; bz < blockSize.z ; ++bz ) {
							auto thread = block->thread(bx, by, bz);
							assert(thread.pos() == dim3(bx, by, bz));
						}
					}
				}
			}
		}
	}
	for (size_t i=0 ; i<grid.blockCount() ; ++i) {
		assert(grid.block(i));
	}
}

static void test_device() {
	int count = -1;
	int driverVersion = -1;
	cu_assert(cuDriverGetVersion(&driverVersion));
	assert(driverVersion > 0);
	cu_assert(cuDeviceGetCount(&count));
	assert(count > 0);
	CUdevice devId = 0;
	cu_assert(cuDeviceGet(&devId, 0));
	char name[512];
	cu_assert(cuDeviceGetName(name,512,devId));
	size_t totalMemory=0;
	cu_assert(cuDeviceTotalMem(&totalMemory,devId));
	assert(totalMemory > 0);
	std::cout << "using device: " << name << " with " << (totalMemory/(1024.0f * 1024)) << " mb memory\n";
}

static void test_memory(){
	int to_test = 5, to_get = -1;
	CUdeviceptr dev_value;
	cu_assert(cuMemAlloc(&dev_value,sizeof(to_test)));
	cu_assert(cuMemcpyHtoD(dev_value, &to_test, sizeof(to_test)));
	cu_assert(cuMemcpyDtoH(&to_get, dev_value, sizeof(to_get)));
	assert(to_get == to_test);
	cu_assert(cuMemFree(dev_value));
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
	CUdeviceptr devValue;
	int hostValue = 10;
	cu_assert(cuMemAlloc(&devValue, sizeof(int)));
	cu_assert(cuMemcpyHtoD(devValue, &hostValue, sizeof(hostValue)));
	void * params[] = {&devValue};
	assert(hostValue != 5);
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr));
	cu_assert(cuMemcpyDtoH(&hostValue, devValue, sizeof(hostValue)));
	assert(hostValue == 5);
	cu_assert(cuMemFree(devValue));
}

void test_cuda(){
	std::cout << "testing cuda...\n";
	cu_assert(cuInit(0));
	test_grid();
	test_device();
	test_memory();
	test_module();
	std::cout << "done.\n";
}
