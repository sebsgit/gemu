#include "drivers/cuda/cuda.h"
#include "../arch/Device.h"
#include "cuda/cudaThreads.h"
#include "debug/KernelDebugger.h"
#include "VariableDeclaration.h"
#include "Load.h"
#include "Convert.h"
#include "Move.h"
#include "Store.h"
#include "Return.h"
#include <cassert>
#include <iostream>

#define cu_assert(x) (assert((x) == CUDA_SUCCESS))

static void test_grid(){
	using namespace gemu::cuda;
	const dim3 gridSize(10, 10, 2);
	const dim3 blockSize(32, 32, 5);
	ThreadGrid grid(gridSize, blockSize);
	assert(grid.blockCount() == gridSize.x * gridSize.y * gridSize.z);
	assert(grid.block(0,0,0));
	assert(grid.block(0,0,0)->size() == blockSize);
	assert(grid.block(gridSize.x + 1, 0, 0) == nullptr);
	for (size_t gx = 0 ; gx < gridSize.x ; ++gx) {
		for (size_t gy = 0 ; gy < gridSize.y ; ++gy) {
			for (size_t gz = 0 ; gz < gridSize.z ; ++gz) {
				auto block = grid.block(gx, gy, gz);
				assert(block);
				assert(block->size() == blockSize);
				assert(block->pos() == dim3(gx, gy, gz));
				assert(block->threadCount() == blockSize.x * blockSize.y * blockSize.z);
				for (size_t bx = 0 ; bx < blockSize.x ; ++bx) {
					for (size_t by = 0 ; by < blockSize.y ; ++by) {
						for (size_t bz = 0 ; bz < blockSize.z ; ++bz ) {
							auto thread = block->thread(bx, by, bz);
							assert(thread.pos() == dim3(bx, by, bz));
						}
					}
				}
			}
		}
	}
	for (size_t i=0 ; i<grid.blockCount() ; ++i) {
		assert(grid.block(i));
	}
}

static void test_device() {
	int count = -1;
	int driverVersion = -1;
	cu_assert(cuDriverGetVersion(&driverVersion));
	assert(driverVersion > 0);
	cu_assert(cuDeviceGetCount(&count));
	assert(count > 0);
	CUdevice devId = 0;
	cu_assert(cuDeviceGet(&devId, 0));
	char name[512];
	cu_assert(cuDeviceGetName(name,512,devId));
	size_t totalMemory=0;
	cu_assert(cuDeviceTotalMem(&totalMemory,devId));
	assert(totalMemory > 0);
	std::cout << "using device: " << name << " with " << (totalMemory/(1024.0f * 1024)) << " mb memory\n";
}

static void test_memory(){
	int to_test = 5, to_get = -1;
	CUdeviceptr dev_value;
	cu_assert(cuMemAlloc(&dev_value,sizeof(to_test)));
	cu_assert(cuMemcpyHtoD(dev_value, &to_test, sizeof(to_test)));
	cu_assert(cuMemcpyDtoH(&to_get, dev_value, sizeof(to_get)));
	assert(to_get == to_test);
	cu_assert(cuMemFree(dev_value));
}

static void test_module() {
	const std::string test_source =
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry kernel(.param .u64 kernel_param_0) {\n"
	".reg .s32 	%r<2>;\n"
	".reg .s64 	%rd<3>;\n"
	"ld.param.u64 	%rd1, [kernel_param_0];\n"
	"cvta.to.global.u64 	%rd2, %rd1;\n"
	"mov.u32 	%r1, -5;\n"
	"st.global.u32 	[%rd2], %r1;\n"
	"ret;\n"
	"}\n";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, test_source.c_str()));
	assert(cuModuleGetFunction(&funcHandle, modId, "nosuchkernel___123") == CUDA_ERROR_NOT_FOUND);
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel"));
	CUdeviceptr devValue;
	int hostValue = 10;
	cu_assert(cuMemAlloc(&devValue, sizeof(int)));
	cu_assert(cuMemcpyHtoD(devValue, &hostValue, sizeof(hostValue)));
	void * params[] = {&devValue};
	assert(hostValue != 5);
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr));
	cu_assert(cuMemcpyDtoH(&hostValue, devValue, sizeof(hostValue)));
	assert(hostValue == -5);
	cu_assert(cuMemFree(devValue));
	cu_assert(cuModuleUnload(modId));
}

static void test_debugger() {
#ifdef PTX_KERNEL_DEBUG
    const std::string test_source =
    ".version 4.2\n"
    ".target sm_20\n"
    ".address_size 64\n"
    ".visible .entry kernel(.param .u64 kernel_param_0) {\n"
    ".reg .s32 	%r<2>;\n"
    ".reg .s64 	%rd<3>;\n"
    "ld.param.u64 	%rd1, [kernel_param_0];\n"
    "cvta.to.global.u64 	%rd2, %rd1;\n"
    "mov.u32 	%r1, -5;\n"
    "st.global.u32 	[%rd2], %r1;\n"
    "ret;\n"
    "}\n";
    ptx::debug::KernelDebugger debugger;
    CUmodule modId = 0;
    CUfunction funcHandle = 0;
    cu_assert(cuModuleLoadData(&modId, test_source.c_str()));
    assert(cuModuleGetFunction(&funcHandle, modId, "nosuchkernel___123") == CUDA_ERROR_NOT_FOUND);
    cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel"));
    CUdeviceptr devValue;
    int hostValue = 10;
    cu_assert(cuMemAlloc(&devValue, sizeof(int)));
    cu_assert(cuMemcpyHtoD(devValue, &hostValue, sizeof(hostValue)));
    void * params[] = {&devValue};
    assert(hostValue != 5);
    cu_assert(cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr));
    assert(std::dynamic_pointer_cast<ptx::VariableDeclaration>(debugger.step()));
    assert(std::dynamic_pointer_cast<ptx::VariableDeclaration>(debugger.step()));
    assert(std::dynamic_pointer_cast<ptx::Load>(debugger.step()));
    assert(std::dynamic_pointer_cast<ptx::Convert>(debugger.step()));
    assert(std::dynamic_pointer_cast<ptx::Move>(debugger.step()));
    assert(std::dynamic_pointer_cast<ptx::Store>(debugger.step()));
	assert(ptx::param_cast<int>(debugger.symbols()["%r1"]) == -5);
    assert(std::dynamic_pointer_cast<ptx::Return>(debugger.step()));
    assert(debugger.step().get() == nullptr);
    cu_assert(cuMemcpyDtoH(&hostValue, devValue, sizeof(hostValue)));
    assert(hostValue == -5);
    cu_assert(cuMemFree(devValue));
    cu_assert(cuModuleUnload(modId));
#endif
}

static void test_module_2() {
	const std::string source = ".visible .entry kernel_3(\n"
	".param .u32 kernel_3_param_0,\n"
	".param .u64 kernel_3_param_1\n"
	")\n"
	"{\n"
	".reg .s32 	%r<2>;\n"
	".reg .s64 	%rd<3>;\n"
	"ld.param.u32 	%r1, [kernel_3_param_0];\n"
	"ld.param.u64 	%rd1, [kernel_3_param_1];\n"
	"cvta.to.global.u64 	%rd2, %rd1;\n"
	"st.global.u32 	[%rd2], %r1;\n"
	"ret;\n"
	"}";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel_3"));
	CUdeviceptr devValue;
	int hostValue = 10;
	cu_assert(cuMemAlloc(&devValue, sizeof(int)));
	void * params[] = {&hostValue, &devValue};
	cu_assert(cuLaunchKernel(funcHandle, 2,5,3, 8,3,2, 0,0, params, nullptr));
	int result = 0;
	cu_assert(cuMemcpyDtoH(&result, devValue, sizeof(result)));
	assert(result == hostValue);
	cu_assert(cuMemFree(devValue));
	cu_assert(cuModuleUnload(modId));
}

/*
__global__ void kernel_4(const int in, int * out){
	*out = in + 7;
}
*/
static void test_module_with_add() {
	const std::string source = ".visible .entry kernel_4(\n"
	".param .u32 kernel_4_param_0,\n"
	".param .u64 kernel_4_param_1\n"
	")\n"
	"{\n"
	".reg .s32 	%r<3>;\n"
	".reg .s64 	%rd<3>;\n"
	"ld.param.u32 	%r1, [kernel_4_param_0];\n"
	"ld.param.u64 	%rd1, [kernel_4_param_1];\n"
	"cvta.to.global.u64 	%rd2, %rd1;\n"
	"add.s32 	%r2, %r1, 7;\n"
	"st.global.u32 	[%rd2], %r2;\n"
	"ret;\n"
	"}";

	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel_4"));
	CUdeviceptr devValue;
	int hostValue = 10;
	cu_assert(cuMemAlloc(&devValue, sizeof(int)));
	void * params[] = {&hostValue, &devValue};
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr));
	int result = 0;
	cu_assert(cuMemcpyDtoH(&result, devValue, sizeof(result)));
	assert(result == hostValue + 7);
	cu_assert(cuMemFree(devValue));
	cu_assert(cuModuleUnload(modId));
}

/*
__global__ void kernel_4(const int in, int * out){
	*out = in * 31;
}
*/
static void test_module_with_mul() {
	const std::string source = ".visible .entry kernel_4(\n"
	".param .u32 kernel_4_param_0,\n"
	".param .u64 kernel_4_param_1\n"
	")\n"
	"{\n"
	".reg .s32 	%r<3>;\n"
	".reg .s64 	%rd<3>;\n"
	"ld.param.u32 	%r1, [kernel_4_param_0];\n"
	"ld.param.u64 	%rd1, [kernel_4_param_1];\n"
	"cvta.to.global.u64 	%rd2, %rd1;\n"
	"mul.wide.s32 	%r2, %r1, 31;\n"
	"st.global.u32 	[%rd2], %r2;\n"
	"ret;\n"
	"}";

	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel_4"));
	CUdeviceptr devValue;
	int hostValue = 10;
	cu_assert(cuMemAlloc(&devValue, sizeof(int)));
	void * params[] = {&hostValue, &devValue};
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr));
	int result = 0;
	cu_assert(cuMemcpyDtoH(&result, devValue, sizeof(result)));
	assert(result == hostValue * 31);
	cu_assert(cuMemFree(devValue));
	cu_assert(cuModuleUnload(modId));
}

static void test_module_with_branch() {
	const std::string test_source =
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry kernel(.param .u64 kernel_param_0) {\n"
	".reg .s32 	%r<2>;\n"
	".reg .s64 	%rd<3>;\n"
	"bra 	BB1_2;\n"
	"ld.param.u64 	%rd1, [kernel_param_0];\n"
	"cvta.to.global.u64 	%rd2, %rd1;\n"
	"mov.u32 	%r1, 5;\n"
	"st.global.u32 	[%rd2], %r1;\n"
	"BB1_2: ret;\n"
	"}\n";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, test_source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel"));
	CUdeviceptr devValue;
	int hostValue = 10;
	cu_assert(cuMemAlloc(&devValue, sizeof(int)));
	cu_assert(cuMemcpyHtoD(devValue, &hostValue, sizeof(hostValue)));
	void * params[] = {&devValue};
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr));
	cu_assert(cuMemcpyDtoH(&hostValue, devValue, sizeof(hostValue)));
	assert(hostValue == 10);
	cu_assert(cuMemFree(devValue));
	cu_assert(cuModuleUnload(modId));
}

static void test_module_with_arithm() {
	const std::string source = ".visible .entry kernel( .param .u64 kernel_2_param_0, .param .u64 kernel_2_param_1 ){\n"
	".reg .pred 	%p<2>;\n"
	".reg .s32 	%r<5>;\n"
	".reg .s64 	%rd<7>;\n"
	"ld.param.u64 	%rd1, [kernel_2_param_0];\n"
	"ld.param.u64 	%rd2, [kernel_2_param_1];\n"
	"cvta.to.global.u64 	%rd3, %rd2;\n"
	"ldu.global.u32 	%r2, [%rd3];\n"
	"mov.u32 	%r1, %tid.x;\n"
	"setp.ge.u32	%p1, %r1, %r2;\n"
	"@%p1 bra 	BB1_2;\n"
	"cvta.to.global.u64 	%rd4, %rd1;\n"
	"mul.wide.u32 	%rd5, %r1, 4;\n"
	"add.s64 	%rd6, %rd4, %rd5;\n"
	"ld.global.u32 	%r3, [%rd6];\n"
	"shl.b32 	%r4, %r3, 1;\n"
	"st.global.u32 	[%rd6], %r4;\n"
"BB1_2:\n"
	"ret;\n }";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel"));
}

static void test_module_with_sync() {
	const std::string source =
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry kernel(\n"
	".param .u64 _Z6kernelPii_param_0,\n"
	".param .u32 _Z6kernelPii_param_1 ) {\n"
	".reg .pred 	%p<5>;\n"
	".reg .s32 	%r<19>;\n"
	".reg .s64 	%rd<3>;\n"
	".shared .u32 _Z6kernelPii$__cuda_local_var_41819_30_non_const_counter;\n"
	"ld.param.u64 	%rd1, [_Z6kernelPii_param_0];\n"
	"ld.param.u32 	%r7, [_Z6kernelPii_param_1];\n"
	"mov.u32 	%r1, %tid.x;\n"
	"setp.ne.s32	%p1, %r1, 0;\n"
	"@%p1 bra 	BB0_2;\n"
	"mov.u32 	%r8, 0;\n"
	"st.shared.u32 	[_Z6kernelPii$__cuda_local_var_41819_30_non_const_counter], %r8;\n"
	"BB0_2:\n"
	"bar.sync 	0;\n"
	"mov.u32 	%r17, 0;\n"
	"mov.u32 	%r18, %r17;\n"
	"mov.u32 	%r15, %r17;\n"
	"setp.lt.s32	%p2, %r7, 1;\n"
	"@%p2 bra 	BB0_4;\n"
	"BB0_3:\n"
	"add.s32 	%r18, %r15, %r18;\n"
	"add.s32 	%r15, %r15, 1;\n"
	"setp.lt.s32	%p3, %r15, %r7;\n"
	"mov.u32 	%r17, %r18;\n"
	"@%p3 bra 	BB0_3;\n"
	"BB0_4:\n"
	"bar.sync 	0;\n"
	"ld.shared.u32 	%r12, [_Z6kernelPii$__cuda_local_var_41819_30_non_const_counter];\n"
	"add.s32 	%r13, %r12, %r17;\n"
	"st.shared.u32 	[_Z6kernelPii$__cuda_local_var_41819_30_non_const_counter], %r13;\n"
	"bar.sync 	0;\n"
	"@%p1 bra 	BB0_6;\n"
	"cvta.to.global.u64 	%rd2, %rd1;\n"
	"ld.shared.u32 	%r14, [_Z6kernelPii$__cuda_local_var_41819_30_non_const_counter];\n"
	"st.global.u32 	[%rd2], %r14;\n"
	"BB0_6:\n"
	"ret;\n"
	"}";
	CUmodule modId = 0;
	CUfunction funcHandle = 0;
	cu_assert(cuModuleLoadData(&modId, source.c_str()));
	cu_assert(cuModuleGetFunction(&funcHandle, modId, "kernel"));
	CUdeviceptr devResult;
	cu_assert(cuMemAlloc(&devResult, sizeof(int)));
	int count = 12;
	void * params[] = {&devResult, &count};
	cu_assert(cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr));
	int result = 0;
	cu_assert(cuMemcpyDtoH(&result, devResult, sizeof(result)));
	cu_assert(cuMemFree(devResult));
	cu_assert(cuModuleUnload(modId));
    assert(result == 66);
}

static void test_module_with_kernel_array() {
    /*

    __global__ void kernel(int* out, int* in, int count) {
        int v[256];
        for (int i=0 ; i<256 ; ++i)
            v[i] = 0;
        for (int i=0 ; i<count ; ++i)
            v[ in[i] % 256 ] += 1;
        for (int i=0 ; i<256 ; ++i)
            out[i] = v[i];
    }
    *
    */

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
    "	// .globl	_Z6kernelPiS_i\n"
    "\n"
    ".visible .entry _Z6kernelPiS_i(\n"
    "	.param .u64 _Z6kernelPiS_i_param_0,\n"
    "	.param .u64 _Z6kernelPiS_i_param_1,\n"
    "	.param .u32 _Z6kernelPiS_i_param_2\n"
    ")\n"
    "{\n"
    "	.local .align 4 .b8 	__local_depot0[1024];\n" //TODO declare this correctly
    "	.reg .b64 	%SP;\n"
    "	.reg .b64 	%SPL;\n"
    "	.reg .pred 	%p<5>;\n"
    "	.reg .b32 	%r<39>;\n"
    "	.reg .b64 	%rd<25>;\n"
    "\n"
    "\n"
    "	mov.u64 	%rd24, __local_depot0;\n"
    "	cvta.local.u64 	%SP, %rd24;\n"
    "	ld.param.u64 	%rd13, [_Z6kernelPiS_i_param_0];\n"
    "	ld.param.u64 	%rd12, [_Z6kernelPiS_i_param_1];\n"
    "	ld.param.u32 	%r7, [_Z6kernelPiS_i_param_2];\n"
    "	cvta.to.global.u64 	%rd23, %rd13;\n"
    "	add.u64 	%rd14, %SP, 0;\n"
    "	cvta.to.local.u64 	%rd2, %rd14;\n"
    "	mov.u32 	%r36, -256;\n"
    "	mov.u64 	%rd22, %rd2;\n"
    "\n"
    "BB0_1:\n"
    "	mov.u64 	%rd15, 0;\n"
    "	st.local.u32 	[%rd22+4], %rd15;\n"
    "	st.local.u32 	[%rd22], %rd15;\n"
    "	st.local.u32 	[%rd22+12], %rd15;\n"
    "	st.local.u32 	[%rd22+8], %rd15;\n"
    "	st.local.u32 	[%rd22+20], %rd15;\n"
    "	st.local.u32 	[%rd22+16], %rd15;\n"
    "	st.local.u32 	[%rd22+28], %rd15;\n"
    "	st.local.u32 	[%rd22+24], %rd15;\n"
    "	st.local.u32 	[%rd22+36], %rd15;\n"
    "	st.local.u32 	[%rd22+32], %rd15;\n"
    "	st.local.u32 	[%rd22+44], %rd15;\n"
    "	st.local.u32 	[%rd22+40], %rd15;\n"
    "	st.local.u32 	[%rd22+52], %rd15;\n"
    "	st.local.u32 	[%rd22+48], %rd15;\n"
    "	st.local.u32 	[%rd22+60], %rd15;\n"
    "	st.local.u32 	[%rd22+56], %rd15;\n"
    "	st.local.u32 	[%rd22+68], %rd15;\n"
    "	st.local.u32 	[%rd22+64], %rd15;\n"
    "	st.local.u32 	[%rd22+76], %rd15;\n"
    "	st.local.u32 	[%rd22+72], %rd15;\n"
    "	st.local.u32 	[%rd22+84], %rd15;\n"
    "	st.local.u32 	[%rd22+80], %rd15;\n"
    "	st.local.u32 	[%rd22+92], %rd15;\n"
    "	st.local.u32 	[%rd22+88], %rd15;\n"
    "	st.local.u32 	[%rd22+100], %rd15;\n"
    "	st.local.u32 	[%rd22+96], %rd15;\n"
    "	st.local.u32 	[%rd22+108], %rd15;\n"
    "	st.local.u32 	[%rd22+104], %rd15;\n"
    "	st.local.u32 	[%rd22+116], %rd15;\n"
    "	st.local.u32 	[%rd22+112], %rd15;\n"
    "	st.local.u32 	[%rd22+124], %rd15;\n"
    "	st.local.u32 	[%rd22+120], %rd15;\n"
    "	add.s64 	%rd22, %rd22, 128;\n"
    "	add.s32 	%r36, %r36, 32;\n"
    "	setp.ne.s32	%p1, %r36, 0;\n"
    "	@%p1 bra 	BB0_1;\n"
    "\n"
    "	cvta.to.global.u64 	%rd18, %rd12;\n"
    "	mov.u32 	%r38, -256;\n"
    "	mov.u32 	%r37, 0;\n"
    "	setp.lt.s32	%p2, %r7, 1;\n"
    "	mov.u64 	%rd21, %rd2;\n"
    "	@%p2 bra 	BB0_4;\n"
    "\n"
    "BB0_3:\n"
    "	ldu.global.u32 	%r12, [%rd18];\n"
    "	shr.s32 	%r13, %r12, 31;\n"
    "	shr.u32 	%r14, %r13, 24;\n"
    "	add.s32 	%r15, %r12, %r14;\n"
    "	and.b32  	%r16, %r15, -256;\n"
    "	sub.s32 	%r17, %r12, %r16;\n"
    "	mul.wide.s32 	%rd16, %r17, 4;\n"
    "	add.s64 	%rd17, %rd2, %rd16;\n"
    "	ld.local.u32 	%r18, [%rd17];\n"
    "	add.s32 	%r19, %r18, 1;\n"
    "	st.local.u32 	[%rd17], %r19;\n"
    "	add.s64 	%rd18, %rd18, 4;\n"
    "	add.s32 	%r37, %r37, 1;\n"
    "	setp.lt.s32	%p3, %r37, %r7;\n"
    "	mov.u64 	%rd20, %rd2;\n"
    "	mov.u64 	%rd21, %rd20;\n"
    "	@%p3 bra 	BB0_3;\n"
    "\n"
    "BB0_4:\n"
    "	ld.local.u32 	%r20, [%rd21];\n"
    "	ld.local.u32 	%r21, [%rd21+4];\n"
    "	ld.local.u32 	%r22, [%rd21+8];\n"
    "	ld.local.u32 	%r23, [%rd21+12];\n"
    "	ld.local.u32 	%r24, [%rd21+16];\n"
    "	ld.local.u32 	%r25, [%rd21+20];\n"
    "	ld.local.u32 	%r26, [%rd21+24];\n"
    "	ld.local.u32 	%r27, [%rd21+28];\n"
    "	ld.local.u32 	%r28, [%rd21+32];\n"
    "	ld.local.u32 	%r29, [%rd21+36];\n"
    "	ld.local.u32 	%r30, [%rd21+40];\n"
    "	ld.local.u32 	%r31, [%rd21+44];\n"
    "	ld.local.u32 	%r32, [%rd21+48];\n"
    "	ld.local.u32 	%r33, [%rd21+52];\n"
    "	ld.local.u32 	%r34, [%rd21+56];\n"
    "	ld.local.u32 	%r35, [%rd21+60];\n"
    "	st.global.u32 	[%rd23], %r20;\n"
    "	st.global.u32 	[%rd23+4], %r21;\n"
    "	st.global.u32 	[%rd23+8], %r22;\n"
    "	st.global.u32 	[%rd23+12], %r23;\n"
    "	st.global.u32 	[%rd23+16], %r24;\n"
    "	st.global.u32 	[%rd23+20], %r25;\n"
    "	st.global.u32 	[%rd23+24], %r26;\n"
    "	st.global.u32 	[%rd23+28], %r27;\n"
    "	st.global.u32 	[%rd23+32], %r28;\n"
    "	st.global.u32 	[%rd23+36], %r29;\n"
    "	st.global.u32 	[%rd23+40], %r30;\n"
    "	st.global.u32 	[%rd23+44], %r31;\n"
    "	st.global.u32 	[%rd23+48], %r32;\n"
    "	st.global.u32 	[%rd23+52], %r33;\n"
    "	st.global.u32 	[%rd23+56], %r34;\n"
    "	st.global.u32 	[%rd23+60], %r35;\n"
    "	add.s64 	%rd23, %rd23, 64;\n"
    "	add.s64 	%rd21, %rd21, 64;\n"
    "	add.s32 	%r38, %r38, 16;\n"
    "	setp.ne.s32	%p4, %r38, 0;\n"
    "	@%p4 bra 	BB0_4;\n"
    "\n"
    "	ret;\n"
    "}\n";

//	#ifdef PTX_KERNEL_DEBUG
//	ptx::debug::KernelDebugger debugger;
//	#endif

    CUmodule modId = 0;
    CUfunction funcHandle = 0;
    cu_assert(cuModuleLoadData(&modId, source.c_str()));
    cu_assert(cuModuleGetFunction(&funcHandle, modId, "_Z6kernelPiS_i"));
	int count = 32;
	int in[count], out[256];
    for (int i=0 ; i<count ; ++i)
        in[i] = i%6;
    CUdeviceptr devIn, devOut, devCount;
    cu_assert(cuMemAlloc(&devCount, sizeof(count)));
	cu_assert(cuMemAlloc(&devIn, sizeof(in[0]) * count));
	cu_assert(cuMemAlloc(&devOut, sizeof(out[0]) * 256));
    cu_assert(cuMemcpyHtoD(devCount, &count, sizeof(count)));
	cu_assert(cuMemcpyHtoD(devIn, &in, sizeof(in[0]) * count));
    void * params[] = {&devOut, &devIn, (void*)&count};

    cu_assert(cuLaunchKernel(funcHandle, 1,1,1, 1,1,1, 0,0, params, nullptr));

//	#ifdef PTX_KERNEL_DEBUG
//	while (auto i = debugger.step()) {
//		std::cout << i->toString() << ' ' << ptx::param_cast<unsigned long long>(debugger.symbols()["%rd22"]) << '\n';
//	}
//	#endif

	::memset(out, 0, sizeof(out[0]) * 256);
	cu_assert(cuMemcpyDtoH(&out, devOut, sizeof(out[0]) * 256));
	for (int i=0 ; i<256 ; ++i)
        std::cout << out[i] << ' ';
	cu_assert(cuMemFree(devOut));
	cu_assert(cuMemFree(devIn));
	cu_assert(cuMemFree(devCount));
}

static void test_events() {
    CUevent startEvent, endEvent;
    cu_assert(cuEventCreate(&startEvent, CU_EVENT_DEFAULT));
    cu_assert(cuEventCreate(&endEvent, CU_EVENT_DEFAULT));
    cu_assert(cuEventDestroy(startEvent));
    cu_assert(cuEventDestroy(endEvent));
}

static void test_modules() {
    test_module();
    test_module_2();
    test_module_with_branch();
    test_module_with_add();
    test_module_with_mul();
    test_module_with_arithm();
    test_module_with_sync();
	//test_module_with_kernel_array();
}

void test_cuda(){
	std::cout << "testing cuda...\n";
	cu_assert(cuInit(0));
	test_grid();
	test_device();
	test_memory();
	test_modules();
    test_events();
    test_debugger();
	std::cout << "done.\n";
}
