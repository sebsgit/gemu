#ifndef PTXSEMANTINTBITANDH
#define PTXSEMANTINTBITANDH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class BitAnd : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		BitAnd(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<bit_and> " + MemoryInstruction::toString();
		}
		void resolve(SymbolTable& symbols) const override {
            param_storage_t result;
			result.data = symbols.get(this->_operands[1].symbol()).data & symbols.get(this->_operands[2].symbol()).data;
			symbols.set(this->_operands[0].symbol(), result);
		}
	};
}

#endif
