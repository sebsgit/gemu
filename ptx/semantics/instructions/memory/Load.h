#ifndef PTXSEMANTICLOADINSTRH
#define PTXSEMANTICLOADINSTRH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class Load : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		Load(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<load> " + MemoryInstruction::toString();
		}
	};
}

#endif
