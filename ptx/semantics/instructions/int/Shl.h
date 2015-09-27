#ifndef PTXSEMANTINTSHLLH
#define PTXSEMANTINTSHLLH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class Shl : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		Shl(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<shl> " + MemoryInstruction::toString();
		}
	};
}

#endif
