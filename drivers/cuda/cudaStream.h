#ifndef GEMUCUDASTREAM_H
#define GEMUCUDASTREAM_H

#include "cudaDefines.h"
#include "cudaThreads.h"
#include "semantics/Function.h"
#include "runtime/PtxExecutionContext.h"

namespace gemu {
namespace cuda {
    class Stream {
    public:
        Stream(gemu::Device& device, CUstream streamId);
        CUresult launch(CUfunction f,
                  const dim3& gridDim,
                  const dim3& blockDim,
                  unsigned int sharedMemBytes,
                  void** kernelParams,
                  void** extra);
        void synchronize();
    private:
        const CUstream _streamId;
        gemu::Device& _device;
        gemu::cuda::ThreadGrid * _grid = nullptr;
        int _currentBlock = -1;
    };
}
}

#endif // CUDASTREAM_H

