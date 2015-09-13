#ifndef PTXSEMANTICCONVERTINSTRH
#define PTXSEMANTICCONVERTINSTRH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class Convert : public MemoryInstruction {
	public:
		Convert(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<convert> " + MemoryInstruction::toString();
		}
	};
}

#endif
