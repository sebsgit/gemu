#include "test_base.h"

/*
 * __global__ void kernel(bool a, bool b, bool * output) {
        output[0] = a || b;
        output[1] = a && b;
        output[2] = !a || b;
        output[3] = a && !b;
}
 * */

static void launch_kernel(bool a, bool b) {
	const std::string source = 
	".version 4.3\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry _Z6kernelbbPb(\n"
	"	.param .u8 _Z6kernelbbPb_param_0,\n"
	"	.param .u8 _Z6kernelbbPb_param_1,\n"
	"	.param .u64 _Z6kernelbbPb_param_2\n"
	")\n"
	"{\n"
	"	.reg .pred 	%p<7>;\n"
	"	.reg .b16 	%rs<11>;\n"
	"	.reg .b64 	%rd<3>;\n"
	"\n"
	"\n"
	"	ld.param.u64 	%rd1, [_Z6kernelbbPb_param_2];\n"
	"	cvta.to.global.u64 	%rd2, %rd1;\n"
	"	ld.param.s8 	%rs1, [_Z6kernelbbPb_param_0];\n"
	"	ld.param.s8 	%rs2, [_Z6kernelbbPb_param_1];\n"
	"	or.b16  	%rs3, %rs2, %rs1;\n"
	"	and.b16  	%rs4, %rs3, 255;\n"
	"	setp.ne.s16	%p1, %rs4, 0;\n"
	"	selp.u16	%rs5, 1, 0, %p1;\n"
	"	st.global.u8 	[%rd2], %rs5;\n"
	"	and.b16  	%rs6, %rs2, 255;\n"
	"	setp.ne.s16	%p2, %rs6, 0;\n"
	"	and.b16  	%rs7, %rs1, 255;\n"
	"	setp.ne.s16	%p3, %rs7, 0;\n"
	"	and.pred  	%p4, %p2, %p3;\n"
	"	selp.u16	%rs8, 1, 0, %p4;\n"
	"	st.global.u8 	[%rd2+1], %rs8;\n"
	"	setp.eq.s16	%p5, %rs6, 0;\n"
	"	and.pred  	%p6, %p3, %p5;\n"
	"	selp.u16	%rs9, 1, 0, %p6;\n"
	"	xor.b16  	%rs10, %rs9, 1;\n"
	"	st.global.u8 	[%rd2+2], %rs10;\n"
	"	st.global.u8 	[%rd2+3], %rs9;\n"
	"	ret;\n"
	"}\n"
	"\n"
	"\n"
	;
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "_Z6kernelbbPb"));
	CUdeviceptr devOutput;
	cu_assert(cuMemAlloc(&devOutput, 4 * sizeof(bool)));
	void * params[] = {&a, &b, &devOutput};
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr));
	bool result[4] = {true, true, true, true};
	cu_assert(cuMemcpyDtoH(result, devOutput, 4 * sizeof(bool)));
	for (int i=0 ; i<4 ; ++i)
		std::cout << result[i] << "\n";
	cu_assert(cuMemFree(devOutput));
	cu_assert(cuModuleUnload(modId));
	std::cout << "\n";
}

int main(){
	init_test();
	launch_kernel(true, false);
	launch_kernel(false, false);
	launch_kernel(false, true);
	launch_kernel(true, true);
	return 0;
}
