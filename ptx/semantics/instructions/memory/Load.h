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
			param_storage_t source = symbols.get(this->_operands[1]);
			if (this->space() == AllocSpace::Global) {
				memcpy(&stored.data, (void*)(source.data + this->_operands[1].offset()), this->size());
			} else {
				memcpy(&stored, &source, this->size());
			}
			symbols.set(this->_operands[0], stored);
        }
	};
}

#endif
