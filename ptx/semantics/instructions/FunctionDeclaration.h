#ifndef PTXINSTRFUNCDECLH
#define PTXINSTRFUNCDECLH

#include "semantics/Instruction.h"
#include "semantics/Function.h"

namespace ptx {
	class FunctionDeclaration : public Instruction {
		PTX_DECLARE_DISPATCH
	public:
		FunctionDeclaration(const ptx::Function& toDecl)
			:_toDeclare(toDecl)
		{
		}
		ptx::Function func() const {
			return this->_toDeclare;
		}
		std::string toString() const override {
			return "<declare function> " + this->_toDeclare.name();
		}
	private:
		ptx::Function _toDeclare;
	};
}

#endif
