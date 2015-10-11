#include "test_base.h"
#include <cstring>

int main(){
	init_test();
	const std::string source =
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry _Z6kernelPiii("
	"	.param .u64 _Z6kernelPiii_param_0,"
	"	.param .u32 _Z6kernelPiii_param_1,"
	"	.param .u32 _Z6kernelPiii_param_2"
	")"
	"{"
	"	.reg .pred 	%p<4>;"
	"	.reg .s32 	%r<6>;"
	"	.reg .s64 	%rd<5>;"
	""
	""
	"	ld.param.u64 	%rd1, [_Z6kernelPiii_param_0];"
	"	ld.param.u32 	%r3, [_Z6kernelPiii_param_1];"
	"	ld.param.u32 	%r4, [_Z6kernelPiii_param_2];"
	"	mov.u32 	%r1, %tid.x;"
	"	setp.lt.u32	%p1, %r1, %r3;"
	"	mov.u32 	%r2, %tid.y;"
	"	setp.lt.u32	%p2, %r2, %r4;"
	"	and.pred  	%p3, %p1, %p2;"
	"	@!%p3 bra 	BB0_2;"
	"	bra.uni 	BB0_1;"
	""
	"BB0_1:"
	"	cvta.to.global.u64 	%rd2, %rd1;"
	"	mad.lo.s32 	%r5, %r2, %r3, %r1;"
	"	mul.wide.s32 	%rd3, %r5, 4;"
	"	add.s64 	%rd4, %rd2, %rd3;"
	"	st.global.u32 	[%rd4], %r5;"
	""
	"BB0_2:"
	"	ret;"
	"}";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "_Z6kernelPiii"));
	unsigned width = 32;
	unsigned height = 32;
	const unsigned size = width * height;
    int *result;
    result = (int*)malloc(size * sizeof(result[0]));
    memset(result, 0, size * sizeof(result[0]));
    CUdeviceptr resultDev;
	cu_assert(cuMemAlloc(&resultDev, sizeof(result[0]) * size));
	void * params[] = {&resultDev, &width, &height};
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, width,height,1, 0,0, params, nullptr));
	cu_assert(cuMemcpyDtoH(result, resultDev, sizeof(result[0]) * size));
    for (unsigned i=0 ; i<height ; ++i) {
		for (unsigned j=0 ; j<width ; ++j)
			std::cout << result[i*width+j] << ' ';
		std::cout << "\n";
	}
	cu_assert(cuMemFree(resultDev));
	free(result);
	cu_assert(cuModuleUnload(modId));
	return 0;
}
