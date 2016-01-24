#ifndef GEMUCUDASTREAM_H
#define GEMUCUDASTREAM_H

#include "cudaDefines.h"
#include "cudaThreads.h"
#include "semantics/Function.h"
#include "runtime/PtxExecutionContext.h"
#include "cudaEvent.h"
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
        KernelLaunchItem(const ThreadGrid& grid, const ptx::Function& kernel, const ptx::SymbolTable& symbols)
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
    class StreamCallbackItem : public AbstractStreamItem {
    public:
        StreamCallbackItem(CUstream stream, CUstreamCallback callback, void* userData)
            :_stream(stream)
            ,_callback(callback)
            ,_userData(userData)
        {

        }
        void execute() override;
    private:
        CUstream _stream;
        CUstreamCallback _callback;
        void* _userData;
    };
    class EventRecordItem : public AbstractStreamItem {
    public:
        EventRecordItem(Event* event)
            :_event(event)
        {
        }
        void execute() override;
    private:
        Event* _event;
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
        CUresult addCallback(CUstream stream,
                             CUstreamCallback callback,
                             void* userData);
        CUresult recordEvent(Event* event);
        CUresult waitForEvent(Event* event);
        void synchronize();
        unsigned int flags() const { return this->_flags; }
        CUresult status();
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
    private:
	Stream(const Stream&);
	Stream(Stream&&);
	Stream& operator=(const Stream&);
	Stream& operator=(Stream&&);
    };
}
}

#endif // CUDASTREAM_H

