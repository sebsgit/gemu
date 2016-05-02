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
			auto dest = symbols.get(this->_operands[0]);
			const auto source = symbols.get(this->_operands[1]);
			if (symbols.variable(this->_operands[1]).space()==AllocSpace::Shared){
				param_cast<unsigned long long>(dest) = symbols.address(this->_operands[1]);
			} else {
				dest = source;
			}
			symbols.set(this->_operands[0], dest);
        }
	};
}

#endif
