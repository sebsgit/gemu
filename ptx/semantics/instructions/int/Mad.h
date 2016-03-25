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
	        const param_storage_t left = symbols.get(this->_operands[1].symbol());
	        const param_storage_t right = symbols.get(this->_operands[2].symbol());
			const param_storage_t add = symbols.get(this->_operands[3].symbol());
			param_storage_t dest = computeOperator<Multiplication>(this->type(),this->size(),left,right);
			dest = computeOperator<Addition>(this->type(), this->size(), dest, add);
	        symbols.set(this->_operands[0].symbol(), dest);
		}
	};
}

#endif
