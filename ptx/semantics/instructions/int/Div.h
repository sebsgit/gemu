#ifndef PTXSEMANTINTDIVLH
#define PTXSEMANTINTDIVLH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
    class Div : public MemoryInstruction {
        PTX_DECLARE_DISPATCH
    public:
        Div(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
        std::string toString() const override {
            return "<div> " + MemoryInstruction::toString();
        }
        void resolve(SymbolTable& symbols) const override {
            this->dispatchArithmetic<Division>(symbols);
        }
    };
}

#endif
