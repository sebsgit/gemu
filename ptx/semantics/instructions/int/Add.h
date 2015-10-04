#ifndef PTXSEMANTINTADDH
#define PTXSEMANTINTADDH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class Add : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		Add(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<add> " + MemoryInstruction::toString();
		}
		void resolve(SymbolTable& symbols) const override {
            this->dispatchArithmetic<Addition>(symbols);
		}
	};
}

#endif
