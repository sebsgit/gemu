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
            param_storage_t dest = symbols.get(this->_operands[0].symbol());
            param_storage_t source;
            const std::string srcName = this->_operands[1].symbol();
            if (symbols.has(srcName)) {
                source = symbols.get(srcName);
            } else {
                //TODO: LITERAL
                source.data = atoi(srcName.c_str());
            }
            memcpy(&dest, &source, this->size());
            symbols.set(this->_operands[0].symbol(), dest);
        }
	};
}

#endif
