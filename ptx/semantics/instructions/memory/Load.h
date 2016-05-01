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
			auto source = symbols.get(this->_operands[1]);
			if (this->space() == AllocSpace::Global) {
				param_copy_from(stored, (void*)(param_cast<address_t>(source) + this->_operands[1].offset()), this->size());
			} else {
				stored = source;
			}
			symbols.set(this->_operands[0], stored);
        }
	};
}

#endif
