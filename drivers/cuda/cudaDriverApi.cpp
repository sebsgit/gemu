#include "cudaDriverApi.h"

gemu::cuda::GlobalContext * _driverContext = nullptr;
gemu::Device * _default_cuda_device = nullptr;
const CUdevice _default_cuda_device_id = 1;
gemu::cuda::Stream * _default_cuda_stream = nullptr;

CUresult cuInit (unsigned int flags) {
	if (!_default_cuda_device) {
		_default_cuda_device = new gemu::Device(1024 * 1024 * 1024);
		_driverContext = new gemu::cuda::GlobalContext();
        _default_cuda_stream = new gemu::cuda::Stream(*_default_cuda_device,0);
	}
	return CUDA_SUCCESS;
}

CUresult cuCtxCreate(CUcontext*, unsigned int, CUdevice) {
	return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext) {
	return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion (int* driverVersion) {
	*driverVersion = 1;
	return CUDA_SUCCESS;
}

CUresult cuDeviceGet ( CUdevice* device, int  ordinal ) {
	if (ordinal == 0)
		*device = _default_cuda_device_id;
	return ordinal==0 ? CUDA_SUCCESS : CUDA_ERROR_NO_DEVICE;
}
CUresult cuDeviceGetAttribute ( int* pi, CUdevice_attribute attrib, CUdevice dev ) {
	return CUDA_SUCCESS;
}
CUresult cuDeviceGetCount ( int* count ) {
	*count = 1;
	return CUDA_SUCCESS;
}
CUresult cuDeviceGetName ( char* name, int  len, CUdevice dev ) {
	if (dev == _default_cuda_device_id) {
		snprintf(name, len, "gemu_default_cuda_device");
		return CUDA_SUCCESS;
	}
	return CUDA_ERROR_UNKNOWN;
}
CUresult cuDeviceTotalMem ( size_t* bytes, CUdevice dev ) {
	if (dev == _default_cuda_device_id)
		*bytes = _default_cuda_device->memory()->maximumSize();
	return dev == _default_cuda_device_id ? CUDA_SUCCESS : CUDA_ERROR_NO_DEVICE;
}



CUresult cuMemAlloc ( CUdeviceptr* dptr, size_t bytesize ) {
	void * memory = _default_cuda_device->memory()->alloc(bytesize);
	if (memory)
		*dptr = reinterpret_cast<CUdeviceptr>(memory);
	return memory ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult cuMemFree ( CUdeviceptr dptr ) {
	return _default_cuda_device->memory()->free(dptr) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemcpyDtoH ( void* dstHost, CUdeviceptr srcDevice, size_t byteCount ) {
	if (_default_cuda_device->memory()->isValid(srcDevice) && (memcpy(dstHost, (const void *)srcDevice, byteCount) != 0)) {
		return CUDA_SUCCESS;
	}
	return CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemcpyHtoD ( CUdeviceptr dstDevice, const void* srcHost, size_t byteCount ) {
	if (_default_cuda_device->memory()->isValid(dstDevice) && (memcpy(dstDevice, srcHost, byteCount) != 0)) {
		return CUDA_SUCCESS;
	}
	return CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemAllocHost ( void** dptr, size_t bytesize ) {
    void * memory = _default_cuda_device->memory()->allocLocked(bytesize);
    if (memory)
        *dptr = reinterpret_cast<CUdeviceptr>(memory);
    return memory ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult cuMemFreeHost ( void* dptr ) {
    return _default_cuda_device->memory()->freeLocked(dptr) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemAlloc_v2 ( CUdeviceptr* dptr, size_t bytesize ) { return cuMemAlloc(dptr, bytesize); }
CUresult cuMemFree_v2 ( CUdeviceptr dptr ) { return cuMemFree(dptr); }
CUresult cuMemcpyDtoH_v2 ( void* dstHost, CUdeviceptr srcDevice, size_t byteCount ) { return cuMemcpyDtoH(dstHost,srcDevice,byteCount); }
CUresult cuMemcpyHtoD_v2 ( CUdeviceptr dstDevice, const void* srcHost, size_t byteCount ) { return cuMemcpyHtoD(dstDevice,srcHost,byteCount); }
CUresult cuMemAllocHost_v2 ( CUdeviceptr* dptr, size_t bytesize ) { return cuMemAllocHost(dptr, bytesize); }
