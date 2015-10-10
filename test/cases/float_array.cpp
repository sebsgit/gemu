#include "test_base.h"
#include <cstring>

/*
__global__ void kernel(int * out, const int size) {
	if (threadIdx.x < size)
		out[threadIdx.x] = threadIdx.x;
}
*/

int main(){
	init_test();
	const std::string test_source =
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry _Z6kernelPfi(\n"
	".param .u64 _Z6kernelPfi_param_0,\n"
	".param .u32 _Z6kernelPfi_param_1){\n"
	".reg .pred 	%p<2>;\n"
	".reg .f32 	%f<3>;\n"
	".reg .s32 	%r<3>;\n"
	".reg .s64 	%rd<5>;\n"
	"ld.param.u64 	%rd1, [_Z6kernelPfi_param_0];\n"
	"ld.param.u32 	%r2, [_Z6kernelPfi_param_1];\n"
	"mov.u32 	%r1, %tid.x;\n"
	"setp.ge.u32	%p1, %r1, %r2;\n"
	"@%p1 bra 	BB0_2;\n"
	"cvta.to.global.u64 	%rd2, %rd1;\n"
	"cvt.rn.f32.u32	%f1, %r1;\n"
	"mul.f32 	%f2, %f1, 0f3FC00000;\n"
	"mul.wide.u32 	%rd3, %r1, 4;\n"
	"add.s64 	%rd4, %rd2, %rd3;\n"
	"st.global.f32 	[%rd4], %f2;\n"
	"BB0_2:\n"
	"ret;\n"
	"}";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, test_source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "_Z6kernelPfi"));
	CUdeviceptr devArray;
	int size = 10;
	float hostArray[size];
	memset(hostArray, 0, size * sizeof(hostArray[0]));
	cu_assert(cuMemAlloc(&devArray, sizeof(float) * size));
	void * params[] = {&devArray, &size};
	auto result = cuLaunchKernel(funcHandle, 1,1,1, size*2,1,1, 0,0, params, nullptr);
	cu_assert(result);
	cu_assert(cuMemcpyDtoH(&hostArray, devArray, sizeof(hostArray[0])*size));
	cu_assert(cuMemFree(devArray));
	cu_assert(cuModuleUnload(modId));
	for (int i=0 ; i<size ; ++i)
		std::cout << hostArray[i] << '\n';
	return 0;
}
