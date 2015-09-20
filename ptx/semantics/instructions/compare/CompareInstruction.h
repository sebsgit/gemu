#ifndef PTXCOMPAREINSTRUCTONH
#define PTXCOMPAREINSTRUCTONH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class CompareInstruction : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		CompareInstruction(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
	};
}

#endif
