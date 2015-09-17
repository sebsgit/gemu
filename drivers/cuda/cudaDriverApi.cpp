#ifndef GEMUCUDADRIVERAPIH
#define GEMUCUDADRIVERAPIH

#include "cudaDefines.h"
#include "cudaForward.h"
#include "arch/Device.h"
#include <cstring>

static gemu::Device * _default_cuda_device = nullptr;
static const CUdevice _default_cuda_device_id = 1;

CUresult cuInit (unsigned int flags) {
	if (!_default_cuda_device) {
		_default_cuda_device = new gemu::Device(1024 * 1024 * 1024);
	}
	return CUDA_SUCCESS_;
}

CUresult cuDriverGetVersion (int* driverVersion) {
	*driverVersion = 1;
	return CUDA_SUCCESS_;
}

CUresult cuDeviceGet ( CUdevice* device, int  ordinal ) {
	if (ordinal == 0)
		*device = _default_cuda_device_id;
	return ordinal==0 ? CUDA_SUCCESS_ : CUDA_ERROR_NO_DEVICE_;
}
CUresult cuDeviceGetAttribute ( int* pi, CUdevice_attribute attrib, CUdevice dev ) {
	return CUDA_SUCCESS_;
}
CUresult cuDeviceGetCount ( int* count ) {
	*count = 1;
	return CUDA_SUCCESS_;
}
CUresult cuDeviceGetName ( char* name, int  len, CUdevice dev ) {
	if (dev == _default_cuda_device_id) {
		snprintf(name, len, "gemu_default_cuda_device");
		return CUDA_SUCCESS_;
	}
	return CUDA_ERROR_UNKNOWN_;
}
CUresult cuDeviceTotalMem ( size_t* bytes, CUdevice dev ) {
	if (dev == _default_cuda_device_id)
		*bytes = _default_cuda_device->memory()->maximumSize();
	return dev == _default_cuda_device_id ? CUDA_SUCCESS_ : CUDA_ERROR_NO_DEVICE_;
}



CUresult cuMemAlloc ( CUdeviceptr* dptr, size_t bytesize ) {
	void * memory = _default_cuda_device->memory()->alloc(bytesize);
	if (memory)
		*dptr = reinterpret_cast<CUdeviceptr>(memory);
	return memory ? CUDA_SUCCESS_ : CUDA_ERROR_OUT_OF_MEMORY_;
}

CUresult cuMemFree ( CUdeviceptr dptr ) {
	return _default_cuda_device->memory()->free(dptr) ? CUDA_SUCCESS_ : CUDA_ERROR_INVALID_VALUE_;
}

CUresult cuMemcpyDtoH ( void* dstHost, CUdeviceptr srcDevice, size_t byteCount ) {
	if (_default_cuda_device->memory()->isValid(srcDevice) && (memcpy(dstHost, (const void *)srcDevice, byteCount) != 0)) {
		return CUDA_SUCCESS_;
	}
	return CUDA_ERROR_INVALID_VALUE_;
}

CUresult cuMemcpyHtoD ( CUdeviceptr dstDevice, const void* srcHost, size_t byteCount ) {
	if (_default_cuda_device->memory()->isValid(dstDevice) && (memcpy(dstDevice, srcHost, byteCount) != 0)) {
		return CUDA_SUCCESS_;
	}
	return CUDA_ERROR_INVALID_VALUE_;
}

#include "cuda/cudaDriver_moduleImpl.cpp"

#endif
