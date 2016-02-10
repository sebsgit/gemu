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

KernelDebugger::Result KernelDebugger::status() const {
    return this->_stepResult;
}

void KernelDebugger::waitForLaunch() {
    this->_execStarted.waitFor(true);
}

void KernelDebugger::exec(exec::PtxExecutionContext *context, const InstructionList &list){
    assert(context);
    this->_context = context;
    context->_instr = &list;
    this->_execStarted.set(true);
    while (1){
        this->_singleStep.waitFor(true);
        this->_stepResult = Done;
        if (context->_pc < list.count()) {
            this->_lastInstruction = list.fetch(context->_pc++);
            assert(this->_lastInstruction.get());
            this->_lastInstruction->dispatch(*context);
            this->_stepResult = Running;
        } else {
            this->_lastInstruction.reset();
        }
        this->_stepDone.set(true);
        if (this->_stepResult != Running)
            break;
    }
    if (this->_stepResult != Wait)
        context->_instr = nullptr;
}

InstructionPtr KernelDebugger::step() {
    this->_singleStep.set(true);
    this->_stepDone.waitFor(true);
    return this->_lastInstruction;
}

SymbolTable& KernelDebugger::symbols() {
    assert(this->_context);
    return this->_context->_symbols;
}

}
}

#endif

