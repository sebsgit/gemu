#ifndef PTXSEMANTICLOADINSTRH
#define PTXSEMANTICLOADINSTRH

#include "semantics/Instruction.h"

namespace ptx {
	class Load : public MemoryInstruction {
	public:
		Load(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<load> " + MemoryInstruction::toString();
		}
	};
}

#endif
