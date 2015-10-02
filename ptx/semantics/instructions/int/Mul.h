#ifndef PTXSEMANTINTMULLH
#define PTXSEMANTINTMULLH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class Mul : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		Mul(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<mul> " + MemoryInstruction::toString();
		}
		void resolve(SymbolTable& symbols) const override {
			param_storage_t dest = symbols.get(this->_operands[0].symbol());
			const param_storage_t src1 = symbols.get(this->_operands[1].symbol());
			const param_storage_t src2 = symbols.get(this->_operands[2].symbol());
			switch (this->type()) {
			case Type::Signed:
				dest.i = src1.i * src2.i;
				break;
			case Type::Unsigned:
				dest.u = src1.u * src2.u;
				break;
			case Type::Float:
				dest.f = src1.f * src2.f;
				break;
			default:
				break;
			}
			symbols.set(this->_operands[0].symbol(), dest);
		}
	};
}

#endif
