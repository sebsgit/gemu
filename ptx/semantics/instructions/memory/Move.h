#ifndef PTXSEMANTICMOVEINSTRH
#define PTXSEMANTICMOVEINSTRH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class Move : public MemoryInstruction {
	public:
		Move(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<move> " + MemoryInstruction::toString();
		}
	};
}

#endif
