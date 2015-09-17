#ifndef GEMUCUDAFWDDEFINEDH
#define GEMUCUDAFWDDEFINEDH

#include "cuda/cudaDefines.h"
#include <memory>

#define GEMU_DECLARE_EXTERN extern "C"

GEMU_DECLARE_EXTERN CUresult cuInit (unsigned int);
GEMU_DECLARE_EXTERN CUresult cuDriverGetVersion (int*);
GEMU_DECLARE_EXTERN CUresult cuDeviceGet ( CUdevice*, int );
GEMU_DECLARE_EXTERN CUresult cuDeviceGetAttribute ( int*, CUdevice_attribute, CUdevice );
GEMU_DECLARE_EXTERN CUresult cuDeviceGetCount ( int* );
GEMU_DECLARE_EXTERN CUresult cuDeviceGetName ( char*, int , CUdevice );
GEMU_DECLARE_EXTERN CUresult cuDeviceTotalMem ( size_t*, CUdevice );
GEMU_DECLARE_EXTERN CUresult cuMemAlloc ( CUdeviceptr*, size_t );
GEMU_DECLARE_EXTERN CUresult cuMemFree ( CUdeviceptr );
GEMU_DECLARE_EXTERN CUresult cuMemcpyDtoH ( void*, CUdeviceptr, size_t );
GEMU_DECLARE_EXTERN CUresult cuMemcpyHtoD ( CUdeviceptr, const void*, size_t );


#undef GEMU_DECLARE_EXTERN

#endif
