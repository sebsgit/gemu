#ifndef PTXSEMANTINTREMH
#define PTXSEMANTINTREMH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
    class Rem : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
        Rem(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
            return "<rem> " + MemoryInstruction::toString();
		}
		void resolve(SymbolTable& symbols) const override {
            this->dispatchArithmetic<Remainder>(symbols);
		}
	};
}

#endif
