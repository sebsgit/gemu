#include "test_base.h"

/* 
	__global__ 
	void kernel(float * out, float a, float b, float c){
		*out = a*b+c;
	}

*/

static float launch_kernel(float a, float b, float c){
	const std::string source = 
	"//\n"
	"// Generated by NVIDIA NVVM Compiler\n"
	"//\n"
	"// Compiler Build ID: CL-19856038\n"
	"// Cuda compilation tools, release 7.5, V7.5.17\n"
	"// Based on LLVM 3.4svn\n"
	"//\n"
	"\n"
	".version 4.3\n"
	".target sm_20\n"
	".address_size 64\n"
	"\n"
	"	// .globl	kernel\n"
	"\n"
	".visible .entry kernel(\n"
	"	.param .u64 _Z6kernelPffff_param_0,\n"
	"	.param .f32 _Z6kernelPffff_param_1,\n"
	"	.param .f32 _Z6kernelPffff_param_2,\n"
	"	.param .f32 _Z6kernelPffff_param_3\n"
	")\n"
	"{\n"
	"	.reg .f32 	%f<5>;\n"
	"	.reg .b64 	%rd<3>;\n"
	"\n"
	"\n"
	"	ld.param.u64 	%rd1, [_Z6kernelPffff_param_0];\n"
	"	ld.param.f32 	%f1, [_Z6kernelPffff_param_1];\n"
	"	ld.param.f32 	%f2, [_Z6kernelPffff_param_2];\n"
	"	ld.param.f32 	%f3, [_Z6kernelPffff_param_3];\n"
	"	cvta.to.global.u64 	%rd2, %rd1;\n"
	"	fma.rn.f32 	%f4, %f1, %f2, %f3;\n"
	"	st.global.f32 	[%rd2], %f4;\n"
	"	ret;\n"
	"}\n"
	"\n"
	"\n"
	;
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel"));
	CUdeviceptr devResult;
	cu_assert(cuMemAlloc(&devResult, sizeof(int)));
	void * params[] = {&devResult, &a, &b, &c};
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr));
	float result = 0;
	cu_assert(cuMemcpyDtoH(&result, devResult, sizeof(result)));
	cu_assert(cuMemFree(devResult));
	cu_assert(cuModuleUnload(modId));
	return result;
}

int main(){
	init_test();
	for (float i=1.0f ; i<6.0f ; i += 2.8f)
		for (float j=-4.4f ; j<6.2f ; j += 0.2f)
			for (float k=2.0f ; k<6.0f ; k += 2.0f) {
				float result = launch_kernel(i,j,k);
				std::cout << ((int)(result*10))/10.0f << '\n';
			}
	return 0;
}
