#ifndef KERNELDEBUGGER_H
#define KERNELDEBUGGER_H

#include "PtxExecutionContext.h"
#include <thread>
#include <condition_variable>

#ifdef PTX_KERNEL_DEBUG

namespace ptx {
namespace debug {

class KernelDebugger {
public:
    enum Result {
        Done,
        Wait,
        Running
    };
    KernelDebugger();
    ~KernelDebugger();
    void exec(ptx::exec::PtxExecutionContext* context, const InstructionList& list);
    Result step();
    void waitForLaunch();
private:
    std::mutex _mutex;
    std::condition_variable _waitCondition;
    Result _stepResult = Done;
    bool _execStarted = false;
};


}
}

#endif

#endif // KERNELDEBUGGER_H
