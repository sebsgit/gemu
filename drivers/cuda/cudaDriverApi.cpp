#include "cudaDriverApi.h"
#include "gemuConfig.h"

gemu::cuda::GlobalContext * _driverContext = nullptr;
gemu::Device * _default_cuda_device = nullptr;
const CUdevice _default_cuda_device_id = 1;
gemu::cuda::Stream * _default_cuda_stream = nullptr;

CUresult cuInit (unsigned int flags) {
    PTX_UNUSED(flags);
	if (!_default_cuda_device) {
        _default_cuda_device = new gemu::Device(gemu::config::defaultDevice.totalMemory);
		_driverContext = new gemu::cuda::GlobalContext();
        _default_cuda_stream = new gemu::cuda::Stream(*_default_cuda_device, CU_STREAM_DEFAULT);
	}
	return CUDA_SUCCESS;
}

CUresult cuCtxCreate(CUcontext*, unsigned int, CUdevice) {
	return CUDA_SUCCESS;
}

CUresult cuCtxCreate_v2(CUcontext* context, unsigned int flags, CUdevice device) {
    return cuCtxCreate(context, flags, device);
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
    PTX_UNUSED(pi);
    PTX_UNUSED(attrib);
    PTX_UNUSED(dev);
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
    if (_default_cuda_device->memory()->isValid(srcDevice)) {
        _default_cuda_stream->synchronize();
        return (memcpy(dstHost, (const void *)srcDevice, byteCount) != 0) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
	}
	return CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemcpyHtoD ( CUdeviceptr dstDevice, const void* srcHost, size_t byteCount ) {
	if (_default_cuda_device->memory()->isValid(dstDevice) && (memcpy(dstDevice, srcHost, byteCount) != 0)) {
        _default_cuda_stream->synchronize();
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

CUresult cuStreamCreate(CUstream* stream, unsigned int flags) {
    *stream = _driverContext->createStream(flags);
    return CUDA_SUCCESS;
}

CUresult cuStreamGetFlags (CUstream hStream, unsigned int* flags) {
    if (!flags)
        return CUDA_ERROR_OUT_OF_MEMORY;
    gemu::cuda::Stream* streamPtr = nullptr;
    if (_driverContext->findStream(hStream, &streamPtr)) {
        *flags = streamPtr->flags();
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_HANDLE;
}

CUresult cuStreamDestroy(CUstream stream) {
    return _driverContext->destroyStream(stream) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

CUresult cuStreamSynchronize(CUstream stream) {
    gemu::cuda::Stream* streamPtr = nullptr;
    return _driverContext->findStream(stream, &streamPtr) ? streamPtr->synchronize(), CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

CUresult cuStreamAddCallback( CUstream hStream, CUstreamCallback callback, void* userData, unsigned int  flags ) {
    PTX_UNUSED(flags);
    gemu::cuda::Stream* streamPtr = nullptr;
    return _driverContext->findStream(hStream, &streamPtr) ? streamPtr->addCallback(hStream, callback, userData) : CUDA_ERROR_INVALID_HANDLE;
}

CUresult cuStreamQuery(CUstream stream) {
    gemu::cuda::Stream* streamPtr = nullptr;
    return _driverContext->findStream(stream, &streamPtr) ? streamPtr->status() : CUDA_ERROR_INVALID_HANDLE;
}

CUresult cuEventCreate ( CUevent* phEvent, unsigned int flags ) {
    *phEvent = _driverContext->createEvent(flags);
    return CUDA_SUCCESS;
}

CUresult cuEventDestroy ( CUevent event ) {
    return _driverContext->destroyEvent(event) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

CUresult cuEventElapsedTime ( float* pMilliseconds, CUevent hStart, CUevent hEnd ) {
    gemu::cuda::Event* start=nullptr, *end = nullptr;
    if (_driverContext->findEvent(hStart, &start) && _driverContext->findEvent(hEnd, &end)) {
        *pMilliseconds = start->msTo(*end);
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_HANDLE;
}

CUresult cuEventQuery ( CUevent hEvent ) {
    gemu::cuda::Event* event = nullptr;
    if (_driverContext->findEvent(hEvent, &event) ) {
        return event->wasRecorded() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    }
    return CUDA_ERROR_INVALID_HANDLE;
}

CUresult cuEventRecord ( CUevent hEvent, CUstream stream ) {
    gemu::cuda::Stream* streamPtr = nullptr;
    gemu::cuda::Event* eventPtr = nullptr;
    return (_driverContext->findStream(stream, &streamPtr) && _driverContext->findEvent(hEvent, &eventPtr)) ? eventPtr->setStream(stream), streamPtr->recordEvent(eventPtr) : CUDA_ERROR_INVALID_HANDLE;
}

CUresult cuEventSynchronize ( CUevent hEvent ) {
    gemu::cuda::Event* event = nullptr;
    if (_driverContext->findEvent(hEvent, &event) ) {
        gemu::cuda::Stream* stream = nullptr;
        if (_driverContext->findStream(event->streamId(), &stream)) {
            return stream->waitForEvent(event);
        }
    }
    return CUDA_ERROR_INVALID_HANDLE;
}

CUresult cuMemAlloc_v2 ( CUdeviceptr* dptr, size_t bytesize ) { return cuMemAlloc(dptr, bytesize); }
CUresult cuMemFree_v2 ( CUdeviceptr dptr ) { return cuMemFree(dptr); }
CUresult cuMemcpyDtoH_v2 ( void* dstHost, CUdeviceptr srcDevice, size_t byteCount ) { return cuMemcpyDtoH(dstHost,srcDevice,byteCount); }
CUresult cuMemcpyHtoD_v2 ( CUdeviceptr dstDevice, const void* srcHost, size_t byteCount ) { return cuMemcpyHtoD(dstDevice,srcHost,byteCount); }
CUresult cuMemAllocHost_v2 ( CUdeviceptr* dptr, size_t bytesize ) { return cuMemAllocHost(dptr, bytesize); }
CUresult cuStreamDestroy_v2(CUstream stream) { return cuStreamDestroy(stream); }
CUresult cuEventDestroy_v2(CUevent event) { return cuEventDestroy(event); }
