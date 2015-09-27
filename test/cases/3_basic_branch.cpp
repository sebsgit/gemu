#include "test_base.h"

int main(){
	init_test();
	const std::string source = 
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry kernel(.param .u64 kernel_param_0) {\n"
	".reg .s32 	%r<2>;\n"
	".reg .s64 	%rd<3>;\n"
	"bra 	BB1_2;\n"
	"ld.param.u64 	%rd1, [kernel_param_0];\n"
	"cvta.to.global.u64 	%rd2, %rd1;\n"
	"mov.u32 	%r1, 5;\n"
	"st.global.u32 	[%rd2], %r1;\n"
	"BB1_2: ret;\n"
	"}\n";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel"));
	CUdeviceptr devValue;
	int hostValue = 10;
	cu_assert(cuMemAlloc(&devValue, sizeof(int)));
	cu_assert(cuMemcpyHtoD(devValue, &hostValue, sizeof(hostValue)));
	void * params[] = {&devValue};
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr));
	cu_assert(cuMemcpyDtoH(&hostValue, devValue, sizeof(hostValue)));
	assert(hostValue == 10);
	std::cout << hostValue << "\n";
	cu_assert(cuMemFree(devValue));
	cu_assert(cuModuleUnload(modId));
	return 0;
}
