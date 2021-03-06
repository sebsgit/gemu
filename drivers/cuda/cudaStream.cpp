#include "cudaStream.h"
#include "cudaDriverApi.h"
#include "gemuConfig.h"
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

void StreamCallbackItem::execute() {
    this->_callback(this->_stream, CUDA_SUCCESS, this->_userData);
}

void EventRecordItem::execute() {
    this->_event->record();
}

Stream::Stream(gemu::Device& device, unsigned int flags)
    :_device(device)
    ,_flags(flags)
{
    this->_thread = new std::thread([this](){ this->threadFunction(); });
}

Stream::~Stream(){
    this->_abortThread = true;
    this->_waitCondition.notify_one();
    this->_thread->detach();
    delete this->_thread;
}

inline bool check_dev_caps(const gemu::config::device_t& dev, const dim3& grid, const dim3& block) {
    if( (int32_t)grid.x <= dev.attribs.maxGridDimX && (int32_t)grid.y <= dev.attribs.maxGridDimY && (int32_t)grid.z <= dev.attribs.maxGridDimZ
            && (int32_t)block.x <= dev.attribs.maxBlockDimX && (int32_t)block.y <= dev.attribs.maxBlockDimY && (int32_t)block.z <= dev.attribs.maxBlockDimZ)
    {
        return (int32_t)block.product() <= dev.attribs.maxThreadsPerBlock;
    }
    return false;
}

CUresult Stream::launch(CUfunction f,
          const dim3& gridDim,
          const dim3& blockDim,
          unsigned int sharedMemBytes,
          void** kernelParams,
          void** extra)
{
    (void)sharedMemBytes;
    if (kernelParams && extra)
        return CUDA_ERROR_INVALID_VALUE;
    if (!kernelParams)
        return CUDA_ERROR_INVALID_VALUE;
    const auto deviceCaps = gemu::config::defaultDevice;
    if (!check_dev_caps(deviceCaps, gridDim, blockDim))
        return CUDA_ERROR_INVALID_VALUE;
    auto kernel = _driverContext->function(f);
    if (kernel.isNull())
        return CUDA_ERROR_NOT_FOUND;
    auto funcParams = kernel.parameters();
    ptx::SymbolTable symbols;
    symbols.setGlobalSection(_driverContext->globalSymbols());
	for (size_t i=0 ; i<funcParams.size() ; ++i)
		symbols.set(funcParams[i], ptx::param_storage_t::fromRawData(kernelParams[i], funcParams[i].size()));
    auto kernelLaunch = AbstractStreamItemPtr(new KernelLaunchItem(gemu::cuda::ThreadGrid(gridDim,blockDim),
                                               kernel,
                                               symbols));
    this->_mutex.lock();
    this->_queue.push(kernelLaunch);
    this->_mutex.unlock();
    this->_waitCondition.notify_one();
    return CUDA_SUCCESS;
}

CUresult Stream::addCallback(CUstream stream, CUstreamCallback callback, void *userData){
    this->_mutex.lock();
    this->_queue.push(AbstractStreamItemPtr(
                          new StreamCallbackItem(stream, callback, userData)));
    this->_mutex.unlock();
    this->_waitCondition.notify_one();
    return CUDA_SUCCESS;
}

CUresult Stream::recordEvent(Event *event){
    this->_mutex.lock();
    this->_queue.push(AbstractStreamItemPtr(new EventRecordItem(event)));
    this->_mutex.unlock();
    this->_waitCondition.notify_one();
    return CUDA_SUCCESS;
}

CUresult Stream::waitForEvent(Event *event) {
    while (1) {
        std::unique_lock<std::mutex> lock(this->_mutex);
        if (event->wasRecorded())
            break;
        std::this_thread::yield();
    }
    return CUDA_SUCCESS;
}

void Stream::threadFunction(){
    while (1) {
        std::unique_lock<std::mutex> lock(this->_mutex);
        if (this->_queue.empty()) {
            this->_waitCondition.wait(lock);
        }
        if (this->_abortThread)
            break;
        if (this->_queue.empty()==false) {
            AbstractStreamItemPtr item = this->_queue.front();
            this->_queue.pop();
            this->_working = true;
            lock.unlock();
            item->execute();
            lock.lock();
            this->_working = false;
        }
    }
}

void Stream::synchronize(){
    while (1){
        std::unique_lock<std::mutex> lock(this->_mutex);
        if (this->_queue.empty()==false) {
            lock.unlock();
            std::this_thread::yield();
        } else {
            if (this->_working==false)
                break;
        }
    }
}

CUresult Stream::status() {
    std::unique_lock<std::mutex> lock(this->_mutex);
    if (this->_working || this->_queue.empty()==false)
        return CUDA_ERROR_NOT_READY;
    return CUDA_SUCCESS;
}

} }
