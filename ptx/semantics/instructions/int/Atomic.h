#ifndef PTXSEMANTINTATOMICH
#define PTXSEMANTINTATOMICH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class AtomicAdd : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		AtomicAdd(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<atom add> " + MemoryInstruction::toString();
		}
		void resolve(SymbolTable& symbols) const override {
			const param_storage_t operand2 = symbols.get(this->_operands[2]);
			const unsigned long long address = symbols.get(this->_operands[1]).data;
			param_storage_t * operand1 = reinterpret_cast<param_storage_t*>(address);
			symbols.lockSharedSection();
			symbols.set(this->_operands[0], *operand1);
			operand1->u += operand2.u;
			symbols.unlockSharedSection();
		}
	};
}

#endif
