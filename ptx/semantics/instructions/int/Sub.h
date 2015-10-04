#ifndef PTXSEMANTINTSUBH
#define PTXSEMANTINTSUBH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
    class Sub : public MemoryInstruction {
        PTX_DECLARE_DISPATCH
    public:
        Sub(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
        std::string toString() const override {
            return "<sub> " + MemoryInstruction::toString();
        }
        void resolve(SymbolTable& symbols) const override {
            this->dispatchArithmetic<Subtraction>(symbols);
        }
    };
}

#endif
