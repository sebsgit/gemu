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
            const param_storage_t dest = symbols.get(this->_operands[0].symbol());
            const param_storage_t source = symbols.get(this->_operands[1].symbol());
            *(reinterpret_cast<unsigned int *>(dest.data)) = source.data;
        }
	};
}

#endif
