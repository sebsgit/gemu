#include "test_base.h"

/*
kernel(a, *b)
	if (a != 10)
		*b = 0

*/

static void launch_kernel( int a ) {
	const std::string test_source =
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry kernel(\n"
	".param .u32 _Z6kerneliPi_param_0,\n"
	".param .u64 _Z6kerneliPi_param_1\n"
	"){\n"
	".reg .pred 	%p<2>;\n"
	".reg .s32 	%r<3>;\n"
	".reg .s64 	%rd<3>;\n"
	"ld.param.u32 	%r1, [_Z6kerneliPi_param_0];\n"
	"ld.param.u64 	%rd1, [_Z6kerneliPi_param_1];\n"
	"cvta.to.global.u64 	%rd2, %rd1;\n"
	"setp.eq.s32	%p1, %r1, 10;\n"
	"@!%p1 st.global.u32 	[%rd2], 0;\n"
	"ret;\n"
	"}";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, test_source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel"));
	CUdeviceptr devValue, devValue2;
	int hostValue = a;
	cu_assert(cuMemAlloc(&devValue, sizeof(int)));
	cu_assert(cuMemcpyHtoD(devValue, &hostValue, sizeof(hostValue)));
	void * params[] = {&hostValue, &devValue};
	auto result = cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr);
	cu_assert(result);
	cu_assert(cuMemcpyDtoH(&hostValue, devValue, sizeof(hostValue)));
	cu_assert(cuMemFree(devValue));
	cu_assert(cuModuleUnload(modId));
	std::cout << hostValue << '\n';
}

int main(){
	init_test();
	launch_kernel(10);
	launch_kernel(11);
	return 0;
}
