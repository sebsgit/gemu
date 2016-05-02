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
			const auto source = symbols.get(this->_operands[1]);
			if (this->space() == AllocSpace::Shared || this->space() == AllocSpace::Parameter) {
				symbols.set(this->_operands[0], source);
			} else {
				auto dest = symbols.get(this->_operands[0]);
				source.copyInto(dest, this->size(), this->_operands[0].offset());
			}
        }
	};
}

#endif
