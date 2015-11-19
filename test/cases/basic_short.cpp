#include "test_base.h"

/* __global__ void kernel(short* a, short* b, short* c){
    *c = *a + *b;
}
*/

int main(){
    init_test();
    const std::string source =
        ".version 4.3\n"
        ".target sm_20\n"
        ".address_size 64\n"
        "\n"
        "	// .globl	_Z6kernelPsS_S_\n"
        "\n"
        ".visible .entry kernel(\n"
        "	.param .u64 _Z6kernelPsS_S__param_0,\n"
        "	.param .u64 _Z6kernelPsS_S__param_1,\n"
        "	.param .u64 _Z6kernelPsS_S__param_2\n"
        ")\n"
        "{\n"
        "	.reg .b32 	%r<4>;\n"
        "	.reg .b64 	%rd<7>;\n"
        "\n"
        "\n"
        "	ld.param.u64 	%rd1, [_Z6kernelPsS_S__param_0];\n"
        "	ld.param.u64 	%rd2, [_Z6kernelPsS_S__param_1];\n"
        "	ld.param.u64 	%rd3, [_Z6kernelPsS_S__param_2];\n"
        "	cvta.to.global.u64 	%rd4, %rd3;\n"
        "	cvta.to.global.u64 	%rd5, %rd2;\n"
        "	cvta.to.global.u64 	%rd6, %rd1;\n"
        "	ld.global.u16 	%r1, [%rd6];\n"
        "	ld.global.u16 	%r2, [%rd5];\n"
        "	add.s32 	%r3, %r2, %r1;\n"
        "	st.global.u16 	[%rd4], %r3;\n"
        "	ret;\n"
        "}\n"
        "\n"
        "\n"
    ;
    CUmodule modId = 0;
    CUfunction funcHandle = 0;
    cu_assert(cuModuleLoadData(&modId, source.c_str()));
    cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel"));
    short a = 10, b = 25, c=0;
    CUdeviceptr devA, devB, devC;
    cu_assert(cuMemAlloc(&devA, sizeof(a)));
    cu_assert(cuMemAlloc(&devB, sizeof(b)));
    cu_assert(cuMemAlloc(&devC, sizeof(c)));
    cu_assert(cuMemcpyHtoD(devA, &a, sizeof(a)));
    cu_assert(cuMemcpyHtoD(devB, &b, sizeof(b)));
    void * params[] = {&devA, &devB, &devC};
    cu_assert(cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr));
    cu_assert(cuMemcpyDtoH(&c, devC, sizeof(c)));
    std::cout << c << "\n";
    cu_assert(cuMemFree(devA));
    cu_assert(cuMemFree(devB));
    cu_assert(cuMemFree(devC));
    cu_assert(cuModuleUnload(modId));
    return 0;
}
