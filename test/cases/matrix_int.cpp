#include "test_base.h"
#include <cstring>

int main(){
	init_test();
	const std::string source =
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry _Z6kernelPiS_S_ii("
	"	.param .u64 _Z6kernelPiS_S_ii_param_0,"
	"	.param .u64 _Z6kernelPiS_S_ii_param_1,"
	"	.param .u64 _Z6kernelPiS_S_ii_param_2,"
	"	.param .u32 _Z6kernelPiS_S_ii_param_3,"
	"	.param .u32 _Z6kernelPiS_S_ii_param_4"
	")"
	"{"
	"	.reg .pred 	%p<4>;"
	"	.reg .s32 	%r<9>;"
	"	.reg .s64 	%rd<11>;"
	""
	""
	"	ld.param.u64 	%rd1, [_Z6kernelPiS_S_ii_param_0];"
	"	ld.param.u64 	%rd2, [_Z6kernelPiS_S_ii_param_1];"
	"	ld.param.u64 	%rd3, [_Z6kernelPiS_S_ii_param_2];"
	"	ld.param.u32 	%r3, [_Z6kernelPiS_S_ii_param_3];"
	"	ld.param.u32 	%r4, [_Z6kernelPiS_S_ii_param_4];"
	"	mov.u32 	%r1, %tid.x;"
	"	setp.lt.u32	%p1, %r1, %r3;"
	"	mov.u32 	%r2, %tid.y;"
	"	setp.lt.u32	%p2, %r2, %r4;"
	"	and.pred  	%p3, %p1, %p2;"
	"	@!%p3 bra 	BB0_2;"
	"	bra.uni 	BB0_1;"
	""
	"BB0_1:"
	"	cvta.to.global.u64 	%rd4, %rd2;"
	"	mad.lo.s32 	%r5, %r2, %r3, %r1;"
	"	mul.wide.s32 	%rd5, %r5, 4;"
	"	add.s64 	%rd6, %rd4, %rd5;"
	"	cvta.to.global.u64 	%rd7, %rd3;"
	"	add.s64 	%rd8, %rd7, %rd5;"
	"	ld.global.u32 	%r6, [%rd8];"
	"	ld.global.u32 	%r7, [%rd6];"
	"	add.s32 	%r8, %r6, %r7;"
	"	cvta.to.global.u64 	%rd9, %rd1;"
	"	add.s64 	%rd10, %rd9, %rd5;"
	"	st.global.u32 	[%rd10], %r8;"
	""
	"BB0_2:"
	"	ret;"
	"}";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "_Z6kernelPiS_S_ii"));
	unsigned width = 32;
	unsigned height = 32;
	const unsigned size = width * height;
    int *result, *source1, *source2;
    result = (int*)malloc(size * sizeof(result[0]));
    source1 = (int*)malloc(size * sizeof(result[0]));
    source2 = (int*)malloc(size * sizeof(result[0]));
    memset(result, 0, size * sizeof(result[0]));
    for (unsigned i=0 ; i<size ; ++i) {
		source1[i] = 2 * i;
		source2[i] = 5 * i;
	}
    CUdeviceptr resultDev, source1Dev, source2Dev;
	cu_assert(cuMemAlloc(&resultDev, sizeof(result[0]) * size));
	cu_assert(cuMemAlloc(&source1Dev, sizeof(result[0]) * size));
	cu_assert(cuMemAlloc(&source2Dev, sizeof(result[0]) * size));
	cu_assert(cuMemcpyHtoD(source1Dev, source1, sizeof(result[0]) * size));
	cu_assert(cuMemcpyHtoD(source2Dev, source2, sizeof(result[0]) * size));

	memset(source1, 0, size * sizeof(source1[0]));
	cu_assert(cuMemcpyDtoH(source1, source1Dev, sizeof(result[0]) * size));
	void * params[] = {&resultDev, &source1Dev, &source2Dev, &width, &height};
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, width,height,1, 0,0, params, nullptr));
	cu_assert(cuMemcpyDtoH(result, resultDev, sizeof(result[0]) * size));
    for (unsigned i=0 ; i<height ; ++i) {
		for (unsigned j=0 ; j<width ; ++j)
			std::cout << result[i*width+j] << ' ';
		std::cout << "\n";
	}

	cu_assert(cuMemFree(resultDev));
	cu_assert(cuMemFree(source1Dev));
	cu_assert(cuMemFree(source2Dev));
	free(result);
	free(source1);
	free(source2);
	cu_assert(cuModuleUnload(modId));
	return 0;
}
