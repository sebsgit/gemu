#ifndef GEMUCUDASTREAM_H
#define GEMUCUDASTREAM_H

#include "cudaDefines.h"
#include "cudaThreads.h"
#include "semantics/Function.h"
#include "runtime/PtxExecutionContext.h"
#include <thread>

namespace gemu {
namespace cuda {
    class Stream {
    public:
        Stream(gemu::Device& device, unsigned int flags);
        ~Stream();
        CUresult launch(CUfunction f,
                  const dim3& gridDim,
                  const dim3& blockDim,
                  unsigned int sharedMemBytes,
                  void** kernelParams,
                  void** extra);
        void synchronize();
        unsigned int flags() const { return this->_flags; }
    private:
        void dispatchBlocks();
    private:
        gemu::Device& _device;
        gemu::cuda::ThreadGrid _grid;
        ptx::Function _currentKernel;
        ptx::SymbolTable _currentSymbols;
        std::thread* _thread = nullptr;
        unsigned int _flags = 0;
    };
}
}

#endif // CUDASTREAM_H

