#include "test_base.h"

int main(){
	init_test();
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
	assert(cuModuleGetFunction(&funcHandle, modId, "nosuchkernel___123") == CUDA_ERROR_NOT_FOUND);
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel"));
	CUdeviceptr devValue;
	int hostValue = 10;
	cu_assert(cuMemAlloc(&devValue, sizeof(int)));
	cu_assert(cuMemcpyHtoD(devValue, &hostValue, sizeof(hostValue)));
	void * params[] = {&devValue};
	assert(hostValue != 5);
	auto result = cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr);
	cu_assert(result);
	cu_assert(cuMemcpyDtoH(&hostValue, devValue, sizeof(hostValue)));
	cu_assert(cuMemFree(devValue));
	cu_assert(cuModuleUnload(modId));
	std::cout << hostValue << '\n';
	return 0;
}
