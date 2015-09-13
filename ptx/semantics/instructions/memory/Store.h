#ifndef PTXSEMANTICSTOREINSTRH
#define PTXSEMANTICSTOREINSTRH

#include "semantics/Instruction.h"

namespace ptx {
	class Store : public MemoryInstruction {
	public:
		Store(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<store> " + MemoryInstruction::toString();
		}
	};
}

#endif
