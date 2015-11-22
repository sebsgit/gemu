#include "cudaStream.h"
#include "cudaDriverApi.h"
#include <cassert>

namespace gemu{
namespace cuda{


void KernelLaunchItem::execute() {
    for (size_t i=0 ; i<this->_grid.blockCount() ; ++i) {
        auto block = this->_grid.block(i);
        ptx::exec::PtxBlockDispatcher dispatcher(*_default_cuda_device, *block);
        dispatcher.launch(this->_function, this->_symbols);
        dispatcher.synchronize();
        if (dispatcher.result() != ptx::exec::BlockExecResult::BlockOk){
            std::cout << "block dispatch error.\n";
            break;
        }
    }
}

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
    auto kernel = _driverContext->function(f);
    if (kernel.isNull())
        return CUDA_ERROR_NOT_FOUND;
    auto funcParams = kernel.parameters();
    ptx::SymbolTable symbols;
    for (size_t i=0 ; i<funcParams.size() ; ++i) {
        void * address = kernelParams[i];
        ptx::param_storage_t storage;
        memcpy(&storage, address, funcParams[i].size());
        symbols.set(funcParams[i], storage);
    }
    this->_kernelLaunch = new KernelLaunchItem(gemu::cuda::ThreadGrid(gridDim,blockDim),
                                               kernel,
                                               symbols);
    this->_thread = new std::thread([this](){
        this->_kernelLaunch->execute();
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
