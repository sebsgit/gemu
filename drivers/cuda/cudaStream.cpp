#include "cudaStream.h"
#include "cudaDriverApi.h"
#include <cassert>

namespace gemu{
namespace cuda{

Stream::Stream(gemu::Device& device, unsigned int flags)
    :_device(device)
    ,_flags(flags)
{

}

Stream::~Stream(){
    delete this->_thread;
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
    this->_currentKernel = _driverContext->function(f);
    if (_currentKernel.isNull())
        return CUDA_ERROR_NOT_FOUND;
    this->_grid = gemu::cuda::ThreadGrid(gridDim,blockDim);
    auto funcParams = this->_currentKernel.parameters();
    this->_currentSymbols = ptx::SymbolTable();
    for (size_t i=0 ; i<funcParams.size() ; ++i) {
        void * address = kernelParams[i];
        ptx::param_storage_t storage;
        memcpy(&storage, address, funcParams[i].size());
        this->_currentSymbols.set(funcParams[i], storage);
    }
    this->_thread = new std::thread([this](){
        for (size_t i=0 ; i<this->_grid.blockCount() ; ++i) {
            auto block = this->_grid.block(i);
            ptx::exec::PtxBlockDispatcher dispatcher(*_default_cuda_device, *block);
            dispatcher.launch(this->_currentKernel, this->_currentSymbols);
            dispatcher.synchronize();
            if (dispatcher.result() != ptx::exec::BlockExecResult::BlockOk){
                std::cout << "block dispatch error.\n";
                break;
            }
        }
    });
    return CUDA_SUCCESS;
}

void Stream::synchronize(){
    if (this->_thread) {
        this->_thread->join();
        delete _thread;
        _thread = nullptr;
    }
}

} }
