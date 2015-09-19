#include "ptx/Parser.h"
#include "cudaDriverApi.h"
#include "arch/Device.h"
#include "semantics/Function.h"
#include "cudaThreads.h"

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
	//TODO sainty check on grid size
	ptx::Function func = _driverContext->function(f);
	if (func.isNull())
		return CUDA_ERROR_NOT_FOUND_;
	auto funcParams = func.parameters();
	ptx::exec::SymbolTable symbols;
	for (size_t i=0 ; i<funcParams.size() ; ++i) {
		void * address = kernelParams[i];
		ptx::exec::param_storage_t storage;
		memcpy(&storage, address, funcParams[i].size()/8);
		symbols.set(funcParams[i], storage);
	}
	gemu::cuda::ThreadGrid grid(gemu::cuda::dim3(gridDimX, gridDimY, gridDimZ),
					gemu::cuda::dim3(blockDimX, blockDimY, blockDimZ));
	for (size_t i=0 ; i<grid.blockCount() ; ++i) {
		auto block = grid.block(i);
		ptx::exec::PtxBlockDispatcher dispatcher(*_default_cuda_device, *block);
		if (!dispatcher.launch(func, symbols))
			return CUDA_ERROR_UNKNOWN_;
	}
	return CUDA_SUCCESS_;
}
