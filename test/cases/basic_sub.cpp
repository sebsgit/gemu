#include "test_base.h"

int main(){
	init_test();
	const std::string source = 
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry kernel_4(\n"
	".param .u32 kernel_4_param_0,\n"
	".param .u64 kernel_4_param_1\n"
	")\n"
	"{\n"
	".reg .s32 	%r<3>;\n"
	".reg .s64 	%rd<3>;\n"
	"ld.param.u32 	%r1, [kernel_4_param_0];\n"
	"ld.param.u64 	%rd1, [kernel_4_param_1];\n"
	"cvta.to.global.u64 	%rd2, %rd1;\n"
    "sub.s32 	%r2, %r1, 7;\n"
	"st.global.u32 	[%rd2], %r2;\n"
	"ret;\n"
	"}";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel_4"));
	CUdeviceptr devValue;
    int hostValue = 1;
	cu_assert(cuMemAlloc(&devValue, sizeof(int)));
	void * params[] = {&hostValue, &devValue};
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr));
	int result = 0;
	cu_assert(cuMemcpyDtoH(&result, devValue, sizeof(result)));
    assert(result == hostValue - 7);
	std::cout << result << "\n";
	cu_assert(cuMemFree(devValue));
	cu_assert(cuModuleUnload(modId));
	return 0;
}
