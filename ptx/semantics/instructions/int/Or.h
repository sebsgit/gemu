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
            param_storage_t result;
            result.data = symbols.get(this->_operands[1].symbol()).data | symbols.get(this->_operands[2].symbol()).data;
            symbols.set(this->_operands[0].symbol(), result);
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
            param_storage_t result;
            result.data = symbols.get(this->_operands[1].symbol()).data ^ symbols.get(this->_operands[2].symbol()).data;
            symbols.set(this->_operands[0].symbol(), result);
        }
    };
}

#endif // OR_H

