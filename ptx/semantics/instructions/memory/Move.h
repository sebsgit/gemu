#ifndef PTXSEMANTICMOVEINSTRH
#define PTXSEMANTICMOVEINSTRH

#include "semantics/instructions/memory/MemoryInstruction.h"
#include <cstring>

namespace ptx {
	class Move : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		Move(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<move> " + MemoryInstruction::toString();
		}
        void resolve(SymbolTable& symbols) const override {
			param_storage_t dest = symbols.get(this->_operands[0]);
			const param_storage_t source = symbols.get(this->_operands[1]);
			if (symbols.variable(this->_operands[1]).space()==AllocSpace::Shared){
				dest.data = symbols.address(this->_operands[1]);
			} else {
            	memcpy(&dest, &source, this->size());
			}
			symbols.set(this->_operands[0], dest);
        }
	};
}

#endif
