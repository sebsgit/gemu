#ifndef PTXSEMANTICLOADINSTRH
#define PTXSEMANTICLOADINSTRH

#include "semantics/instructions/memory/MemoryInstruction.h"
#include <cstring>

namespace ptx {
	class Load : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		Load(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<load> " + MemoryInstruction::toString();
		}
        //TODO check with offsets
        void resolve(SymbolTable& symbols) const override {
            param_storage_t stored;
			const auto source = symbols.get(this->_operands[1]);
			if (this->space() == AllocSpace::Global) {
				stored.copyFrom(source, this->size(), this->_operands[1].offset());
			} else {
				stored = source;
			}
			symbols.set(this->_operands[0], stored);
        }
	};
}

#endif
