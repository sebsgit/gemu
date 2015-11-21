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
        Stream(gemu::Device& device, CUstream streamId);
        ~Stream();
        CUresult launch(CUfunction f,
                  const dim3& gridDim,
                  const dim3& blockDim,
                  unsigned int sharedMemBytes,
                  void** kernelParams,
                  void** extra);
        void synchronize();
    private:
        void dispatchBlocks();
    private:
        const CUstream _streamId;
        gemu::Device& _device;
        gemu::cuda::ThreadGrid _grid;
        ptx::Function _currentKernel;
        ptx::SymbolTable _currentSymbols;
        std::thread* _thread = nullptr;
    };
}
}

#endif // CUDASTREAM_H

