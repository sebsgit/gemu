#ifndef PTXSEMANNTICSETPINSTRH
#define PTXSEMANNTICSETPINSTRH

#include "semantics/instructions/compare/CompareInstruction.h"

namespace ptx {
	class Setp : public CompareInstruction {
		PTX_DECLARE_DISPATCH
	public:
		Setp(CompareInstruction&& other) : CompareInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<setp> " + CompareInstruction::toString();
		}
	};
}

#endif
