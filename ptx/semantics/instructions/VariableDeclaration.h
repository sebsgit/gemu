#ifndef PTXINSTRVARDECLH
#define PTXINSTRVARDECLH

#include "semantics/Instruction.h"
#include "semantics/Variable.h"

namespace ptx {
	class VariableDeclaration : public Instruction {
		PTX_DECLARE_DISPATCH
	public:
		VariableDeclaration(const ptx::Variable& toDecl)
			:_toDeclare(toDecl)
		{
		}
		ptx::Variable var() const {
			return this->_toDeclare;
		}
		std::string toString() const override {
			return "<declare> " + this->_toDeclare.name();
		}
	private:
		ptx::Variable _toDeclare;
	};
}

#endif
