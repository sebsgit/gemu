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
	};
}

#endif
