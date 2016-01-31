#include "KernelDebugger.h"
#include "PtxExecutionContext.h"
#include "Instruction.h"

namespace ptx{
namespace debug{

KernelDebugger::KernelDebugger() {
#ifdef PTX_KERNEL_DEBUG
    if (exec::PtxExecutionContext::_debugger) {
        std::cout << "can't have two active debuggers !\n";
        exit(0);
    }
    exec::PtxExecutionContext::_debugger = this;
#endif
}

KernelDebugger::~KernelDebugger() {
    exec::PtxExecutionContext::_debugger = nullptr;
}

void KernelDebugger::exec(exec::PtxExecutionContext *context, const InstructionList &list){
    context->_instr = &list;
    while (context->_pc < list.count()){
        list.fetch(context->_pc++)->dispatch(*context);
        if (context->_barrierWait)
            return;
    }
    context->_instr = nullptr;
}

}
}
