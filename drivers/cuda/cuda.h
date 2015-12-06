#ifndef GEMUCUDAFWDDEFINEDH
#define GEMUCUDAFWDDEFINEDH

#include "cudaDefines.h"
#include <memory>

#define GEMU_DECLARE_EXTERN(F) extern "C" CUresult F ;

GEMU_DECLARE_EXTERN( cuInit (unsigned int) )
GEMU_DECLARE_EXTERN( cuDriverGetVersion (int*) )
GEMU_DECLARE_EXTERN( cuDeviceGet ( CUdevice*, int ) )
GEMU_DECLARE_EXTERN( cuDeviceGetAttribute ( int*, CUdevice_attribute, CUdevice ) )
GEMU_DECLARE_EXTERN( cuDeviceGetCount ( int* ) )
GEMU_DECLARE_EXTERN( cuDeviceGetName ( char*, int , CUdevice ) )
GEMU_DECLARE_EXTERN( cuDeviceTotalMem ( size_t*, CUdevice ) )

GEMU_DECLARE_EXTERN( cuGetErrorName(CUresult, const char**) )
GEMU_DECLARE_EXTERN( cuGetErrorString(CUresult, const char**) )

GEMU_DECLARE_EXTERN( cuCtxCreate(CUcontext*, unsigned int, CUdevice) )
GEMU_DECLARE_EXTERN( cuCtxDestroy(CUcontext) )

GEMU_DECLARE_EXTERN( cuMemAlloc ( CUdeviceptr*, size_t ) )
GEMU_DECLARE_EXTERN( cuMemFree ( CUdeviceptr ) )
GEMU_DECLARE_EXTERN( cuMemcpyDtoH ( void*, CUdeviceptr, size_t ) )
GEMU_DECLARE_EXTERN( cuMemcpyHtoD ( CUdeviceptr, const void*, size_t ) )
GEMU_DECLARE_EXTERN( cuMemAllocHost ( void**, size_t bytesize )  )
GEMU_DECLARE_EXTERN( cuMemFreeHost ( void* ) )

GEMU_DECLARE_EXTERN( cuMemAlloc_v2 ( CUdeviceptr*, size_t ) )
GEMU_DECLARE_EXTERN( cuMemFree_v2 ( CUdeviceptr ) )
GEMU_DECLARE_EXTERN( cuMemcpyDtoH_v2 ( void*, CUdeviceptr, size_t ) )
GEMU_DECLARE_EXTERN( cuMemcpyHtoD_v2 ( CUdeviceptr, const void*, size_t ) )
GEMU_DECLARE_EXTERN( cuMemAllocHost_v2 ( void**, size_t bytesize )  )

GEMU_DECLARE_EXTERN( cuModuleGetFunction ( CUfunction* hfunc, CUmodule hmod, const char* name ) )
GEMU_DECLARE_EXTERN( cuModuleGetGlobal ( CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name ) )
GEMU_DECLARE_EXTERN( cuModuleGetSurfRef ( CUsurfref* pSurfRef, CUmodule hmod, const char* name ) )
GEMU_DECLARE_EXTERN( cuModuleGetTexRef ( CUtexref* pTexRef, CUmodule hmod, const char* name ) )
GEMU_DECLARE_EXTERN( cuModuleLoad ( CUmodule* module, const char* fname ) )
GEMU_DECLARE_EXTERN( cuModuleLoadData ( CUmodule* module, const void* image ) )
// GEMU_DECLARE_EXTERN( cuModuleLoadDataEx ( CUmodule* module, const void* image, unsigned int  numOptions, CUjit_option* options, void** optionValues ) )
GEMU_DECLARE_EXTERN( cuModuleLoadFatBinary ( CUmodule* module, const void* fatCubin ) )
GEMU_DECLARE_EXTERN( cuModuleUnload ( CUmodule hmod ) )

GEMU_DECLARE_EXTERN( cuStreamCreate(CUstream*, unsigned int) )
GEMU_DECLARE_EXTERN( cuStreamDestroy(CUstream) )
GEMU_DECLARE_EXTERN( cuStreamDestroy_v2(CUstream) )
GEMU_DECLARE_EXTERN( cuStreamSynchronize(CUstream) )
GEMU_DECLARE_EXTERN( cuStreamGetFlags (CUstream, unsigned int*) )
GEMU_DECLARE_EXTERN( cuStreamAddCallback ( CUstream hStream, CUstreamCallback callback, void* userData, unsigned int  flags )  )
GEMU_DECLARE_EXTERN( cuStreamQuery(CUstream) )

GEMU_DECLARE_EXTERN( cuLaunchKernel ( CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void** ))

// event management
GEMU_DECLARE_EXTERN( cuEventCreate ( CUevent* phEvent, unsigned int  Flags ) )
GEMU_DECLARE_EXTERN( cuEventDestroy ( CUevent hEvent ) )
GEMU_DECLARE_EXTERN( cuEventDestroy_v2(CUevent) )
GEMU_DECLARE_EXTERN( cuEventElapsedTime ( float* pMilliseconds, CUevent hStart, CUevent hEnd ) )
GEMU_DECLARE_EXTERN( cuEventQuery ( CUevent hEvent ) )
GEMU_DECLARE_EXTERN( cuEventRecord ( CUevent hEvent, CUstream hStream ) )
GEMU_DECLARE_EXTERN( cuEventSynchronize ( CUevent hEvent ) )

#undef GEMU_DECLARE_EXTERN

#endif
