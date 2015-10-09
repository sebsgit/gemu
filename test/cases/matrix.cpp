#include "test_base.h"
#include <cstring>

int main(){
	init_test();
	const std::string source = 
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry matrix_add(\n"
	".param .u64 matrix_add_param_0,\n"
	".param .u64 matrix_add_param_1,\n"
	".param .u64 matrix_add_param_2,\n"
	".param .u32 matrix_add_param_3,\n"
	".param .u32 matrix_add_param_4 ){\n"
	".reg .pred 	%p<4>;\n"
	".reg .f32 	%f<4>;\n"
	".reg .s32 	%r<6>;\n"
	".reg .s64 	%rd<11>;\n"
	"ld.param.u64 	%rd1, [matrix_add_param_0];\n"
	"ld.param.u64 	%rd2, [matrix_add_param_1];\n"
	"ld.param.u64 	%rd3, [matrix_add_param_2];\n"
	"ld.param.u32 	%r4, [matrix_add_param_3];\n"
	"ld.param.u32 	%r3, [matrix_add_param_4];\n"
	"mov.u32 	%r1, %tid.x;\n"
	"mov.u32 	%r2, %tid.y;\n"
	"setp.lt.u32	%p1, %r2, %r3;\n"
	"setp.lt.u32	%p2, %r1, %r4;\n"
	"and.pred  	%p3, %p1, %p2;\n"
	"@!%p3 bra 	BB0_2;\n"
	"bra.uni 	BB0_1;\n"
	"BB0_1:\n"
	"cvta.to.global.u64 	%rd4, %rd2;\n"
	"mad.lo.s32 	%r5, %r1, %r3, %r2;\n"
	"mul.wide.u32 	%rd5, %r5, 4;\n"
	"add.s64 	%rd6, %rd4, %rd5;\n"
	"cvta.to.global.u64 	%rd7, %rd3;\n"
	"add.s64 	%rd8, %rd7, %rd5;\n"
	"ld.global.f32 	%f1, [%rd8];\n"
	"ld.global.f32 	%f2, [%rd6];\n"
	"add.f32 	%f3, %f2, %f1;\n"
	"cvta.to.global.u64 	%rd9, %rd1;\n"
	"add.s64 	%rd10, %rd9, %rd5;\n"
	"st.global.f32 	[%rd10], %f3;\n"
	"BB0_2:\n"
	"ret;\n"
	"}";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "matrix_add"));
	unsigned width = 3;
	unsigned height = 2;
	const unsigned size = width * height;
    float *result, *source1, *source2;
    result = (float*)malloc(size * sizeof(result[0]));
    source1 = (float*)malloc(size * sizeof(result[0]));
    source2 = (float*)malloc(size * sizeof(result[0]));
    memset(result, 0, size * sizeof(result[0]));
    for (unsigned i=0 ; i<size ; ++i) {
		source1[i] = 1.5f * i;
		source2[i] = 2.5f * i;
	}
    CUdeviceptr resultDev, source1Dev, source2Dev;
	cu_assert(cuMemAlloc(&resultDev, sizeof(result[0]) * size));
	cu_assert(cuMemAlloc(&source1Dev, sizeof(result[0]) * size));
	cu_assert(cuMemAlloc(&source2Dev, sizeof(result[0]) * size));
	cu_assert(cuMemcpyHtoD(source1Dev, source1, sizeof(result[0]) * size));
	cu_assert(cuMemcpyHtoD(source2Dev, source2, sizeof(result[0]) * size));
	
	memset(source1, 0, size * sizeof(source1[0]));
	cu_assert(cuMemcpyDtoH(source1, source1Dev, sizeof(result[0]) * size));
	for (unsigned i=0 ; i<size ; ++i)
		std::cout << source1[i] << '\n';
	
	void * params[] = {&resultDev, &source1Dev, &source2Dev, &width, &height};
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, width,height,1, 0,0, params, nullptr));
	cu_assert(cuMemcpyDtoH(result, resultDev, sizeof(result[0]) * size));
    for (unsigned i=0 ; i<size ; ++i)
		std::cout << result[i] << '\n';
	
	cu_assert(cuMemFree(resultDev));
	cu_assert(cuMemFree(source1Dev));
	cu_assert(cuMemFree(source2Dev));
	free(result);
	free(source1);
	free(source2);
	cu_assert(cuModuleUnload(modId));
	return 0;
}
