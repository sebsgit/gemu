#ifndef GEMUCUDASTREAM_H
#define GEMUCUDASTREAM_H

#include "cudaDefines.h"
#include "cudaThreads.h"
#include "semantics/Function.h"
#include "runtime/PtxExecutionContext.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>

namespace gemu {
namespace cuda {

    class AbstractStreamItem {
    public:
        virtual ~AbstractStreamItem(){}
        virtual void execute() = 0;
    };
    typedef std::shared_ptr<AbstractStreamItem> AbstractStreamItemPtr;
    class KernelLaunchItem : public AbstractStreamItem {
    public:
        KernelLaunchItem(const ThreadGrid& grid, ptx::Function kernel, const ptx::SymbolTable& symbols)
            :_grid(grid)
            ,_function(kernel)
            ,_symbols(symbols)
        {

        }
        void execute() override;
    private:
        gemu::cuda::ThreadGrid _grid;
        ptx::Function _function;
        ptx::SymbolTable _symbols;
    };

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
        void threadFunction();
    private:
        gemu::Device& _device;
        std::queue<AbstractStreamItemPtr> _queue;
        std::mutex _mutex;
        std::condition_variable _waitCondition;
        std::thread* _thread = nullptr;
        bool _abortThread = false;
        bool _working = false;
        unsigned int _flags = 0;
    };
}
}

#endif // CUDASTREAM_H

