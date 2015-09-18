#ifndef GEMUCUDADRIVERAPIH
#define GEMUCUDADRIVERAPIH

#include "cudaDefines.h"
#include "cudaForward.h"
#include "arch/Device.h"
#include "semantics/Function.h"
#include <cstring>
#include <vector>
#include <algorithm>

namespace gemu {
	namespace cuda {
		class Module {
		public:
			CUmodule id() const {
				return this->_id;
			}
			void add(const ptx::Function& func) {
				this->_functions.push_back(func);
			}
		private:
			CUmodule _id;
			std::vector<ptx::Function> _functions;
			friend class GlobalContext;
		};
		class GlobalContext {
		public:
			CUmodule add(const Module& m) {
				Module toAdd = m;
				toAdd._id = (CUmodule)++this->_nextModuleId;
				this->_modules.push_back(toAdd);
				return toAdd._id;
			}
			bool remove(CUmodule modId) {
				auto it = std::find_if(this->_modules.begin(),
									   this->_modules.end(),
								   		[&](const Module& m) {return m.id() == modId;});
				if (it != this->_modules.end()) {
					this->_modules.erase(it);
					return true;
				}
				return false;
			}
			CUfunction function(CUmodule modId, const char * name) {
				auto it = std::find_if(this->_modules.begin(),
									   this->_modules.end(),
								   		[&](const Module& m) {return m.id() == modId;});
				if (it != this->_modules.end()) {
					for (size_t i=0 ; i<it->_functions.size() ; ++i) {
						if (it->_functions[i].name() == std::string(name)) {
							auto result = reinterpret_cast<CUfunction>(reinterpret_cast<unsigned long>(modId) * 1000 + i);
							this->_funcCache[result] = it->_functions[i];
							return result;
						}
					}
				}
				return reinterpret_cast<CUfunction>(-1);
			}
			ptx::Function function(CUfunction fhandle) const {
				return this->_funcCache.find(fhandle) != this->_funcCache.end() ? this->_funcCache.find(fhandle)->second : ptx::Function();
			}
		private:
			std::vector<Module> _modules;
			std::unordered_map<CUfunction, ptx::Function> _funcCache;
			unsigned long long _nextModuleId = 0;
		};
	}
}

static gemu::cuda::GlobalContext * _driverContext = nullptr;
static gemu::Device * _default_cuda_device = nullptr;
static const CUdevice _default_cuda_device_id = 1;

CUresult cuInit (unsigned int flags) {
	if (!_default_cuda_device) {
		_default_cuda_device = new gemu::Device(1024 * 1024 * 1024);
		_driverContext = new gemu::cuda::GlobalContext();
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
