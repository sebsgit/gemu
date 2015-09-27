#ifndef GEMUCUDAFWDDEFINEDH
#define GEMUCUDAFWDDEFINEDH

#include "cuda/cudaDefines.h"
#include <memory>

#define GEMU_DECLARE_EXTERN(F) extern "C" CUresult F ;

GEMU_DECLARE_EXTERN( cuInit (unsigned int) )
GEMU_DECLARE_EXTERN( cuDriverGetVersion (int*) )
GEMU_DECLARE_EXTERN( cuDeviceGet ( CUdevice*, int ) )
GEMU_DECLARE_EXTERN( cuDeviceGetAttribute ( int*, CUdevice_attribute, CUdevice ) )
GEMU_DECLARE_EXTERN( cuDeviceGetCount ( int* ) )
GEMU_DECLARE_EXTERN( cuDeviceGetName ( char*, int , CUdevice ) )
GEMU_DECLARE_EXTERN( cuDeviceTotalMem ( size_t*, CUdevice ) )

GEMU_DECLARE_EXTERN( cuCtxCreate(CUcontext*, unsigned int, CUdevice) )
GEMU_DECLARE_EXTERN( cuCtxDestroy(CUcontext) )

GEMU_DECLARE_EXTERN( cuMemAlloc ( CUdeviceptr*, size_t ) )
GEMU_DECLARE_EXTERN( cuMemFree ( CUdeviceptr ) )
GEMU_DECLARE_EXTERN( cuMemcpyDtoH ( void*, CUdeviceptr, size_t ) )
GEMU_DECLARE_EXTERN( cuMemcpyHtoD ( CUdeviceptr, const void*, size_t ) )

GEMU_DECLARE_EXTERN( cuModuleGetFunction ( CUfunction* hfunc, CUmodule hmod, const char* name ) )
GEMU_DECLARE_EXTERN( cuModuleGetGlobal ( CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name ) )
GEMU_DECLARE_EXTERN( cuModuleGetSurfRef ( CUsurfref* pSurfRef, CUmodule hmod, const char* name ) )
GEMU_DECLARE_EXTERN( cuModuleGetTexRef ( CUtexref* pTexRef, CUmodule hmod, const char* name ) )
GEMU_DECLARE_EXTERN( cuModuleLoad ( CUmodule* module, const char* fname ) )
GEMU_DECLARE_EXTERN( cuModuleLoadData ( CUmodule* module, const void* image ) )
// GEMU_DECLARE_EXTERN( cuModuleLoadDataEx ( CUmodule* module, const void* image, unsigned int  numOptions, CUjit_option* options, void** optionValues ) )
GEMU_DECLARE_EXTERN( cuModuleLoadFatBinary ( CUmodule* module, const void* fatCubin ) )
GEMU_DECLARE_EXTERN( cuModuleUnload ( CUmodule hmod ) )

GEMU_DECLARE_EXTERN( cuLaunchKernel ( CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void** ))

#undef GEMU_DECLARE_EXTERN

#endif
