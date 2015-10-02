#include "test_base.h"

int main(){
	init_test();
	const std::string source = 
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry kernel(\n"
	".param .u64 _Z6kernelPff_param_0,\n"
	".param .f32 _Z6kernelPff_param_1)\n"
	"{\n"
	".reg .f32 	%f<3>;\n"
	".reg .s64 	%rd<3>;\n"
	"ld.param.u64 	%rd1, [_Z6kernelPff_param_0];\n"
	"ld.param.f32 	%f1, [_Z6kernelPff_param_1];\n"
	"cvta.to.global.u64 	%rd2, %rd1;\n"
	"add.f32 	%f2, %f1, %f1;\n"
	"st.global.f32 	[%rd2], %f2;\n"
	"ret;\n"
	"}";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel"));
	CUdeviceptr devValue;
	float hostValue = 10;
	cu_assert(cuMemAlloc(&devValue, sizeof(int)));
	void * params[] = {&devValue, &hostValue};
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr));
	float result = 0;
	cu_assert(cuMemcpyDtoH(&result, devValue, sizeof(result)));
	assert_float_eq(result, hostValue * 2);
	std::cout << result << "\n";
	cu_assert(cuMemFree(devValue));
	cu_assert(cuModuleUnload(modId));
	return 0;
}
