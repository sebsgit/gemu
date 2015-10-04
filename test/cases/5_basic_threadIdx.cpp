#include "test_base.h"

/*
 __global__ void kernel(int * out1, int * out2, int * out3) {
	if (threadIdx.x == 0) *out1 = threadIdx.x;
	else if (threadIdx.x == 1) *out2 = threadIdx.x;
	else if (threadIdx.x == 2) *out3 = threadIdx.x;
}
*/

int main(){
	init_test();
	const std::string test_source =
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry kernel(\n"
	".param .u64 _Z6kernelPiS_S__param_0,\n"
	".param .u64 _Z6kernelPiS_S__param_1,\n"
	".param .u64 _Z6kernelPiS_S__param_2) {\n"
	".reg .pred 	%p<4>;\n"
	".reg .s32 	%r<5>;\n"
	".reg .s64 	%rd<7>;\n"
	"ld.param.u64 	%rd1, [_Z6kernelPiS_S__param_0];\n"
	"ld.param.u64 	%rd2, [_Z6kernelPiS_S__param_1];\n"
	"ld.param.u64 	%rd3, [_Z6kernelPiS_S__param_2];\n"
	"mov.u32 	%r1, %tid.x;\n"
	"setp.eq.s32	%p1, %r1, 0;\n"
	"@%p1 bra 	BB0_5;\n"
	"setp.eq.s32	%p2, %r1, 1;\n"
	"@%p2 bra 	BB0_4;\n"
	"bra.uni 	BB0_2;\n"
	"BB0_4:\n"
	"cvta.to.global.u64 	%rd5, %rd2;\n"
	"mov.u32 	%r3, 1;\n"
	"st.global.u32 	[%rd5], %r3;\n"
	"bra.uni 	BB0_6;\n"
	"BB0_5:\n"
	"cvta.to.global.u64 	%rd6, %rd1;\n"
	"mov.u32 	%r4, 0;\n"
	"st.global.u32 	[%rd6], %r4;\n"
	"bra.uni 	BB0_6;\n"
	"BB0_2:\n"
	"setp.ne.s32	%p3, %r1, 2;\n"
	"@%p3 bra 	BB0_6;\n"
	"cvta.to.global.u64 	%rd4, %rd3;\n"
	"mov.u32 	%r2, 2;\n"
	"st.global.u32 	[%rd4], %r2;\n"
	"BB0_6:\n"
	"ret;}";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, test_source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel"));
	CUdeviceptr devValue0, devValue1, devValue2;
	cu_assert(cuMemAlloc(&devValue0, sizeof(int)));
	cu_assert(cuMemAlloc(&devValue1, sizeof(int)));
	cu_assert(cuMemAlloc(&devValue2, sizeof(int)));
	void * params[] = {&devValue0, &devValue1, &devValue2};
	auto result = cuLaunchKernel(funcHandle, 1,1,1, 3,1,1, 0,0, params, nullptr);
	cu_assert(result);
	int hostValue0, hostValue1, hostValue2;
	cu_assert(cuMemcpyDtoH(&hostValue0, devValue0, sizeof(hostValue0)));
	cu_assert(cuMemcpyDtoH(&hostValue1, devValue1, sizeof(hostValue1)));
	cu_assert(cuMemcpyDtoH(&hostValue2, devValue2, sizeof(hostValue2)));
	cu_assert(cuMemFree(devValue0));
	cu_assert(cuMemFree(devValue1));
	cu_assert(cuMemFree(devValue2));
	cu_assert(cuModuleUnload(modId));
	std::cout << hostValue0 << ' ' << hostValue1 << ' ' << hostValue2 << '\n';
	return 0;
}
