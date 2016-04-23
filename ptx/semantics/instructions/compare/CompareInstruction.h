#ifndef PTXCOMPAREINSTRUCTONH
#define PTXCOMPAREINSTRUCTONH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class CompareInstruction : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		void setBoolean(BooleanOperation op){this->_boolOperator = op;}
		void setCompare(CompareOperation cmp){this->_compareOperator = cmp;}
		BooleanOperation booleanOperation() const {return this->_boolOperator;}
		CompareOperation compareOperation() const {return this->_compareOperator;}
	private:
		BooleanOperation _boolOperator = BooleanOperation::NotValidBooleanOperation;
		CompareOperation _compareOperator = CompareOperation::NotValidCompareOperation;
	};
}

#endif
