#ifndef PTXSEMANTINTMADH
#define PTXSEMANTINTMADH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class Mad : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		Mad(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<mad> " + MemoryInstruction::toString();
		}
		void resolve(SymbolTable& symbols) const override {
			const auto left = symbols.get(this->_operands[1]);
			const auto right = symbols.get(this->_operands[2]);
			const auto add = symbols.get(this->_operands[3]);
			auto dest = computeOperator<Multiplication>(this->type(),this->size(),left,right);
			dest = computeOperator<Addition>(this->type(), this->size(), dest, add);
			symbols.set(this->_operands[0], dest);
		}
	};
}

#endif
