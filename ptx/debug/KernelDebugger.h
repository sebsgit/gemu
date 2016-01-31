#ifndef KERNELDEBUGGER_H
#define KERNELDEBUGGER_H

#include "PtxExecutionContext.h"

namespace ptx {
namespace debug {

class KernelDebugger {
public:
    KernelDebugger();
    ~KernelDebugger();
    void exec(ptx::exec::PtxExecutionContext* context, const InstructionList& list);
};

}
}

#endif // KERNELDEBUGGER_H
