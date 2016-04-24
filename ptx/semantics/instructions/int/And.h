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
			const auto result = computeBitwiseOperator<BitwiseAnd>(
						symbols.get(this->_operands[1]),
						symbols.get(this->_operands[2])
					);
			symbols.set(this->_operands[0], result);
		}
	};
}

#endif
