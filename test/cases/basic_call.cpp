#include "test_base.h"

int main(){
	init_test();
	const std::string source = ".version 4.2\n"
   ".target sm_20\n"
   ".address_size 64\n"
   ".visible .func  (.param .b32 func_retval0) _Z4funcii("
   "	.param .b32 _Z4funcii_param_0,"
   "	.param .b32 _Z4funcii_param_1"
   ")"
   "{"
   "	.reg .s32 	%r<4>;"
   ""
   ""
   "	ld.param.u32 	%r1, [_Z4funcii_param_0];"
   "	ld.param.u32 	%r2, [_Z4funcii_param_1];"
   "	add.s32 	%r3, %r2, %r1;"
   "	st.param.b32	[func_retval0+0], %r3;"
   "	ret;"
   "}"
   ".visible .entry _Z6kernelPiii("
   "	.param .u64 _Z6kernelPiii_param_0,"
   "	.param .u32 _Z6kernelPiii_param_1,"
   "	.param .u32 _Z6kernelPiii_param_2"
   ")"
   "{"
   "	.reg .s32 	%r<4>;"
   "	.reg .s64 	%rd<3>;"
   ""
   ""
   "	ld.param.u64 	%rd1, [_Z6kernelPiii_param_0];"
   "	ld.param.u32 	%r1, [_Z6kernelPiii_param_1];"
   "	ld.param.u32 	%r2, [_Z6kernelPiii_param_2];"
   "	cvta.to.global.u64 	%rd2, %rd1;"
   "	{"
   "	.reg .b32 temp_param_reg;"
   "	.param .b32 param0;"
   "	st.param.b32	[param0+0], %r1;"
   "	.param .b32 param1;"
   "	st.param.b32	[param1+0], %r2;"
   "	.param .b32 retval0;"
   "	call.uni (retval0), "
   "	_Z4funcii, "
   "	("
   "	param0, "
   "	param1"
   "	);"
   "	ld.param.b32	%r3, [retval0+0];"
   "	}"
   "	st.global.u32 	[%rd2], %r3;"
   "	ret;"
   "}";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "_Z6kernelPiii"));
	CUdeviceptr devValue;
	int hostValue = 0;
	cu_assert(cuMemAlloc(&devValue, sizeof(int)));
	cu_assert(cuMemcpyHtoD(devValue, &hostValue, sizeof(hostValue)));
	int host1 = 23;
	int host2 = 39;
	void * params[] = {&devValue, &host1, &host2};
	auto result = cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr);
	cu_assert(result);
	cu_assert(cuMemcpyDtoH(&hostValue, devValue, sizeof(hostValue)));
	cu_assert(cuMemFree(devValue));
	cu_assert(cuModuleUnload(modId));
	std::cout << hostValue << '\n';
	return 0;
}
