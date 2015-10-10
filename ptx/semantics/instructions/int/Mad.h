#ifndef PTXSEMANTINTMADH
#define PTXSEMANTINTMADH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class Mad : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		Mad(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<mad> " + MemoryInstruction::toString();
		}
		void resolve(SymbolTable& symbols) const override {
			param_storage_t dest = symbols.get(this->_operands[0].symbol());
	        const param_storage_t left = symbols.get(this->_operands[1].symbol());
	        const param_storage_t right = symbols.get(this->_operands[2].symbol());
			const param_storage_t add = symbols.get(this->_operands[3].symbol());
	        switch (this->type()) {
	        case Type::Signed:
	            dest.i = left.i * right.i + add.i;
	            break;
	        case Type::Unsigned:
	            dest.u = left.u * right.u + add.u;
	            break;
	        default:
	            break;
	        }
	        symbols.set(this->_operands[0].symbol(), dest);
		}
	};
}

#endif
