#include "ptx/Parser.h"

CUresult cuModuleLoadData (CUmodule* module, const void* image) {
	ptx::ParserResult result = ptx::Parser().parseModule(std::string(reinterpret_cast<const char*>(image)));
	if (result.empty()==false) {
		gemu::cuda::Module moduleData;
		for (size_t i=0 ; i<result.count() ; ++i) {
			if (auto fdecl = result.fetch<ptx::FunctionDeclaration>(i)) {
				moduleData.add(fdecl->func());
			}
		}
		*module = _driverContext->add(moduleData);
		return CUDA_SUCCESS_;
	} else {
		return CUDA_ERROR_INVALID_IMAGE_;
	}
}

CUresult cuModuleUnload (CUmodule hmod) {
	return _driverContext->remove(hmod) ? CUDA_SUCCESS_ : CUDA_ERROR_NOT_FOUND_;
}

CUresult cuModuleGetFunction (CUfunction* hfunc, CUmodule hmod, const char* name) {
	CUfunction result = _driverContext->function(hmod, name);
	if (result != reinterpret_cast<CUfunction>(-1)) {
		*hfunc = result;
		return CUDA_SUCCESS_;
	}
	return CUDA_ERROR_NOT_FOUND_;
}

CUresult cuLaunchKernel ( 	CUfunction f,
							unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
							unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
							unsigned int sharedMemBytes,
							CUstream hStream,
							void** kernelParams,
							void** extra )
{
	if (kernelParams && extra)
		return CUDA_ERROR_INVALID_VALUE_;
	return CUDA_ERROR_NOT_SUPPORTED_;
}
