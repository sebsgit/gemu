#include "KernelDebugger.h"
#include "PtxExecutionContext.h"
#include "Instruction.h"
#include <cassert>

#ifdef PTX_KERNEL_DEBUG

namespace ptx{
namespace debug{

KernelDebugger::KernelDebugger() {
    if (exec::PtxExecutionContext::_debugger) {
        std::cout << "can't have two active debuggers !\n";
        exit(0);
    }
    exec::PtxExecutionContext::_debugger = this;
}

KernelDebugger::~KernelDebugger() {
    exec::PtxExecutionContext::_debugger = nullptr;
}

void KernelDebugger::waitForLaunch() {
    std::unique_lock<std::mutex> lock(this->_mutex);
    if (!this->_execStarted)
        this->_waitCondition.wait(lock);
}

void KernelDebugger::exec(exec::PtxExecutionContext *context, const InstructionList &list){
    context->_instr = &list;
    this->_mutex.lock();
    this->_execStarted = true;
    this->_waitCondition.notify_one();
    this->_mutex.unlock();
    while (1){
        std::unique_lock<std::mutex> lock(this->_mutex);
        this->_waitCondition.wait(lock);
        this->_stepResult = Done;
        if (context->_pc < list.count()) {
            InstructionPtr ptr = list.fetch(context->_pc++);
            assert(ptr.get());
            ptr->dispatch(*context);
            this->_stepResult = Running;
        }
        this->_waitCondition.notify_one();
        if (this->_stepResult != Running)
            break;
        if (context->_barrierWait) {
            this->_stepResult = Wait;
            this->_waitCondition.notify_one();
            return;
        }
    }
    context->_instr = nullptr;
}

KernelDebugger::Result KernelDebugger::step() {
    std::unique_lock<std::mutex> lock(this->_mutex);
    this->_waitCondition.notify_one();
    this->_waitCondition.wait(lock);
    return this->_stepResult;
}

}
}

#endif

