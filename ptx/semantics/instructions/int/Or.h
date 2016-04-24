#ifndef PTXSEMANTINTBITORH
#define PTXSEMANTINTBITORH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
    class BitOr : public MemoryInstruction {
        PTX_DECLARE_DISPATCH
    public:
        BitOr(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
        std::string toString() const override {
            return "<bit_or> " + MemoryInstruction::toString();
        }
        void resolve(SymbolTable& symbols) const override {
			const param_storage_t result = computeBitwiseOperator<BitwiseOr>(
						symbols.get(this->_operands[1]),
						symbols.get(this->_operands[2])
					);
			symbols.set(this->_operands[0], result);
        }
    };

    class BitXor : public MemoryInstruction {
        PTX_DECLARE_DISPATCH
    public:
        BitXor(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
        std::string toString() const override {
            return "<bit_xor> " + MemoryInstruction::toString();
        }
        void resolve(SymbolTable& symbols) const override {
			const param_storage_t result = computeBitwiseOperator<BitwiseXOR>(
						symbols.get(this->_operands[1]),
						symbols.get(this->_operands[2])
					);
			symbols.set(this->_operands[0], result);
        }
    };
}

#endif // OR_H

