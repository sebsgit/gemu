#ifndef KERNELDEBUGGER_H
#define KERNELDEBUGGER_H

#include "PtxExecutionContext.h"
#include "Instruction.h"
#include <thread>
#include <condition_variable>

#ifdef PTX_KERNEL_DEBUG

namespace ptx {
namespace debug {

class KernelDebugger {
    class SafeFlag {
    public:
        void waitFor(bool value) {
            std::unique_lock<std::mutex> lock(this->_mutex);
            if (this->_flag != value)
                this->_waitCondition.wait(lock);
            this->_flag = !value;
        }
        void set(bool value) {
            this->_mutex.lock();
            this->_flag = value;
            this->_mutex.unlock();
            this->_waitCondition.notify_one();
        }
    private:
        std::mutex _mutex;
        std::condition_variable _waitCondition;
        bool _flag = false;
    };

public:
    enum Result {
        Done,
        Wait,
        Running,
        NotInitialized
    };
    KernelDebugger();
    ~KernelDebugger();
    void exec(ptx::exec::PtxExecutionContext* context, const InstructionList& list);
    Result status() const;
    InstructionPtr step();
    void waitForLaunch();
    SymbolTable& symbols();
private:
    SafeFlag _execStarted;
    SafeFlag _singleStep;
    SafeFlag _stepDone;
    Result _stepResult = NotInitialized;
    InstructionPtr _lastInstruction;
    ptx::exec::PtxExecutionContext* _context = nullptr;
};


}
}

#endif

#endif // KERNELDEBUGGER_H
