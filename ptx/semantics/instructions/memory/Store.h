#ifndef PTXSEMANTICSTOREINSTRH
#define PTXSEMANTICSTOREINSTRH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class Store : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		Store(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<store> " + MemoryInstruction::toString();
		}
        void resolve(SymbolTable& symbols) const override {
			const param_storage_t source = symbols.get(this->_operands[1]);
			if (this->space() == AllocSpace::Shared || this->space() == AllocSpace::Parameter) {
				symbols.set(this->_operands[0], source);
			} else {
				const param_storage_t dest = symbols.get(this->_operands[0]);
				param_copy_into((void*)(param_cast<address_t>(dest) + this->_operands[0].offset()), source, this->size());
			}
        }
	};
}

#endif
