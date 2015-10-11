#include "test_base.h"

/*  
 
 __global__ void kernel(int * out){
		__shared__ int count;
		if (threadIdx.x==0)
			count=0;
		__syncthreads();
		atomicAdd(&count, 1);
		__syncthreads();
		if (threadIdx.x==0)
			*out = count;
	}
 
 */

int main(){
	init_test();
	const std::string source =
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry _Z6kernelPi("
	"	.param .u64 _Z6kernelPi_param_0"
	")"
	"{"
	"	.reg .pred 	%p<3>;"
	"	.reg .s32 	%r<5>;"
	"	.reg .s64 	%rd<4>;"
	"	.shared .u32 _Z6kernelPi$__cuda_local_var_41819_30_non_const_count;"
	""
	"	ld.param.u64 	%rd1, [_Z6kernelPi_param_0];"
	"	mov.u32 	%r1, %tid.x;"
	"	setp.ne.s32	%p2, %r1, 0;"
	"	@%p2 bra 	BB0_2;"
	""
	"	mov.u32 	%r2, 0;"
	"	st.shared.u32 	[_Z6kernelPi$__cuda_local_var_41819_30_non_const_count], %r2;"
	""
	"BB0_2:"
	"	setp.eq.s32	%p1, %r1, 0;"
	"	bar.sync 	0;"
	"	mov.u64 	%rd2, _Z6kernelPi$__cuda_local_var_41819_30_non_const_count;"
	"	atom.shared.add.u32 	%r3, [%rd2], 1;"
	"	bar.sync 	0;"
	"	@!%p1 bra 	BB0_4;"
	"	bra.uni 	BB0_3;"
	""
	"BB0_3:"
	"	cvta.to.global.u64 	%rd3, %rd1;"
	"	ld.shared.u32 	%r4, [_Z6kernelPi$__cuda_local_var_41819_30_non_const_count];"
	"	st.global.u32 	[%rd3], %r4;"
	""
	"BB0_4:"
	"	ret;"
	"}";

	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "_Z6kernelPi"));
	CUdeviceptr devValue;
    int nthreads = 16;
	cu_assert(cuMemAlloc(&devValue, sizeof(int)));
	void * params[] = {&devValue};
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, nthreads,1,1, 0,0, params, nullptr));
	int result = 0;
	cu_assert(cuMemcpyDtoH(&result, devValue, sizeof(result)));
	std::cout << result << "\n";
	cu_assert(cuMemFree(devValue));
	cu_assert(cuModuleUnload(modId));
	return 0;
}
