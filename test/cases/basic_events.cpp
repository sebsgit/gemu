#include <iostream>
#include <string>
#include "test_base.h"

static int launch_mad_kernel(int a, int b, int c){
	const std::string source = 
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry kernel(\n"
	".param .u64 _Z6kernelPiiii_param_0,\n"
	".param .u32 _Z6kernelPiiii_param_1,\n"
	".param .u32 _Z6kernelPiiii_param_2,\n"
	".param .u32 _Z6kernelPiiii_param_3\n"
	"){\n"
	".reg .s32 	%r<5>;\n"
	".reg .s64 	%rd<3>;\n"
	"ld.param.u64 	%rd1, [_Z6kernelPiiii_param_0];\n"
	"ld.param.u32 	%r1, [_Z6kernelPiiii_param_1];\n"
	"ld.param.u32 	%r2, [_Z6kernelPiiii_param_2];\n"
	"ld.param.u32 	%r3, [_Z6kernelPiiii_param_3];\n"
	"cvta.to.global.u64 	%rd2, %rd1;\n"
	"mad.lo.s32 	%r4, %r2, %r1, %r3;\n"
	"st.global.u32 	[%rd2], %r4;\n"
	"ret;\n"
	"}";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel"));
	CUdeviceptr devResult;
	cu_assert(cuMemAlloc(&devResult, sizeof(int)));
	void * params[] = {&devResult, &a, &b, &c};
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr));
    assert(cuStreamQuery(0) == CUDA_ERROR_NOT_READY);
	int result = 0;
	cu_assert(cuMemcpyDtoH(&result, devResult, sizeof(result)));
	cu_assert(cuMemFree(devResult));
	cu_assert(cuModuleUnload(modId));
	return result;
}

int main(){
	init_test();
	CUevent startEvent, endEvent;
	cu_assert(cuEventCreate(&startEvent, CU_EVENT_DEFAULT));
	cu_assert(cuEventCreate(&endEvent, CU_EVENT_DEFAULT));
	cu_assert(cuEventRecord(startEvent, 0));
	launch_mad_kernel(1,2,3);
	cu_assert(cuEventRecord(endEvent, 0));
	cu_assert(cuEventSynchronize(endEvent));
	cu_assert(cuEventQuery(endEvent));
	float elapsedMs=0.0f;
	cu_assert(cuEventElapsedTime(&elapsedMs, startEvent, endEvent) );
	std::cout << (elapsedMs > 0.0f) << "\n";
	cu_assert(cuEventDestroy(startEvent));
	cu_assert(cuEventDestroy(endEvent));
	return 0;
}
