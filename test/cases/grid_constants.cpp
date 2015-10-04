#include "test_base.h"

/*
 __global__ void kernel(int * blockX, int * blockY, int * blockZ,
					   int * gridX, int * gridY, int * gridZ)
{
	*blockX = blockDim.x;
	*blockY = blockDim.y;
	*blockZ = blockDim.z;
	*gridX = gridDim.x;
	*gridY = gridDim.y;
	*gridZ = gridDim.z;
}
*/

int main(){
	init_test();
	const std::string test_source =
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
    ".visible .entry kernel(\n"
	".param .u64 _Z6kernelPiS_S_S_S_S__param_0,\n"
	".param .u64 _Z6kernelPiS_S_S_S_S__param_1,\n"
	".param .u64 _Z6kernelPiS_S_S_S_S__param_2,\n"
	".param .u64 _Z6kernelPiS_S_S_S_S__param_3,\n"
	".param .u64 _Z6kernelPiS_S_S_S_S__param_4,\n"
	".param .u64 _Z6kernelPiS_S_S_S_S__param_5){\n"
	".reg .s32 	%r<7>;\n"
	".reg .s64 	%rd<13>;\n"
	"ld.param.u64 	%rd1, [_Z6kernelPiS_S_S_S_S__param_0];\n"
	"ld.param.u64 	%rd2, [_Z6kernelPiS_S_S_S_S__param_1];\n"
	"ld.param.u64 	%rd3, [_Z6kernelPiS_S_S_S_S__param_2];\n"
	"ld.param.u64 	%rd4, [_Z6kernelPiS_S_S_S_S__param_3];\n"
	"ld.param.u64 	%rd5, [_Z6kernelPiS_S_S_S_S__param_4];\n"
	"ld.param.u64 	%rd6, [_Z6kernelPiS_S_S_S_S__param_5];\n"
	"cvta.to.global.u64 	%rd7, %rd6;\n"
	"cvta.to.global.u64 	%rd8, %rd5;\n"
	"cvta.to.global.u64 	%rd9, %rd4;\n"
	"cvta.to.global.u64 	%rd10, %rd3;\n"
	"cvta.to.global.u64 	%rd11, %rd2;\n"
	"cvta.to.global.u64 	%rd12, %rd1;\n"
	"mov.u32 	%r1, %ntid.x;\n"
	"st.global.u32 	[%rd12], %r1;\n"
	"mov.u32 	%r2, %ntid.y;\n"
	"st.global.u32 	[%rd11], %r2;\n"
	"mov.u32 	%r3, %ntid.z;\n"
	"st.global.u32 	[%rd10], %r3;\n"
	"mov.u32 	%r4, %nctaid.x;\n"
	"st.global.u32 	[%rd9], %r4;\n"
	"mov.u32 	%r5, %nctaid.y;\n"
	"st.global.u32 	[%rd8], %r5;\n"
	"mov.u32 	%r6, %nctaid.z;\n"
	"st.global.u32 	[%rd7], %r6;\n"
	"ret; }";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, test_source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel"));
	CUdeviceptr devGridX, devGridY, devGridZ;
	CUdeviceptr devBlockX, devBlockY, devBlockZ;
	cu_assert(cuMemAlloc(&devGridX, sizeof(int)));
	cu_assert(cuMemAlloc(&devGridY, sizeof(int)));
	cu_assert(cuMemAlloc(&devGridZ, sizeof(int)));
	cu_assert(cuMemAlloc(&devBlockX, sizeof(int)));
	cu_assert(cuMemAlloc(&devBlockY, sizeof(int)));
	cu_assert(cuMemAlloc(&devBlockZ, sizeof(int)));
	void * params[] = {&devBlockX, &devBlockY, &devBlockZ, &devGridX, &devGridY, &devGridZ};
	cu_assert(cuLaunchKernel(funcHandle, 1,2,3, 6,8,9, 0,0, params, nullptr));
	int values[6];
	cu_assert(cuMemcpyDtoH(&values[0], devBlockX, sizeof(int)));
	cu_assert(cuMemcpyDtoH(&values[1], devBlockY, sizeof(int)));
	cu_assert(cuMemcpyDtoH(&values[2], devBlockZ, sizeof(int)));
	cu_assert(cuMemcpyDtoH(&values[3], devGridX, sizeof(int)));
	cu_assert(cuMemcpyDtoH(&values[4], devGridY, sizeof(int)));
	cu_assert(cuMemcpyDtoH(&values[5], devGridZ, sizeof(int)));
	cu_assert(cuMemFree(devGridX));
	cu_assert(cuMemFree(devGridY));
	cu_assert(cuMemFree(devGridZ));
	cu_assert(cuMemFree(devBlockX));
	cu_assert(cuMemFree(devBlockY));
	cu_assert(cuMemFree(devBlockZ));
	cu_assert(cuModuleUnload(modId));
	for (int i=0 ; i<6 ; ++i)
		std::cout << values[i] << '\n';
	return 0;
}
