#include "cudaStream.h"
#include "cudaDriverApi.h"

namespace gemu{
namespace cuda{

Stream::Stream(gemu::Device& device, CUstream streamId)
    :_streamId(streamId)
    ,_device(device)
{

}

CUresult Stream::launch(CUfunction f,
          const dim3& gridDim,
          const dim3& blockDim,
          unsigned int sharedMemBytes,
          void** kernelParams,
          void** extra)
{
    if (kernelParams && extra)
        return CUDA_ERROR_INVALID_VALUE;
    //TODO sainty check on grid size
    ptx::Function func = _driverContext->function(f);
    if (func.isNull())
        return CUDA_ERROR_NOT_FOUND;
    auto funcParams = func.parameters();
    ptx::SymbolTable symbols;
    for (size_t i=0 ; i<funcParams.size() ; ++i) {
        void * address = kernelParams[i];
        ptx::param_storage_t storage;
        memcpy(&storage, address, funcParams[i].size());
        symbols.set(funcParams[i], storage);
    }
    gemu::cuda::ThreadGrid grid(gridDim,blockDim);
    for (size_t i=0 ; i<grid.blockCount() ; ++i) {
        auto block = grid.block(i);
        ptx::exec::PtxBlockDispatcher dispatcher(*_default_cuda_device, *block);
        dispatcher.launch(func, symbols);
        dispatcher.synchronize();
        if (dispatcher.result() != ptx::exec::BlockExecResult::BlockOk)
            return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}

} }
