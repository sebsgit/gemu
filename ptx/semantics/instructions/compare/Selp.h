#ifndef PTXSEMANTINTSELPH
#define PTXSEMANTINTSELPH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
    class Selp : public MemoryInstruction {
        PTX_DECLARE_DISPATCH
    public:
        Selp(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
        std::string toString() const override {
            return "<selp> " + MemoryInstruction::toString();
        }
        void resolve(SymbolTable& symbols) const override {
			const param_storage_t left = symbols.get(this->_operands[1]);
			const param_storage_t right = symbols.get(this->_operands[2]);
			const param_storage_t condition = symbols.get(this->_operands[3]);
			symbols.set(this->_operands[0], condition.b ? left : right);
        }
    };
}

#endif
