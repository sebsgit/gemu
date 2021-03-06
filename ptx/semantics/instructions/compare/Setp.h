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
        void resolve(SymbolTable& symbols) const override {
            switch (this->compareOperation()) {
            case CompareOperation::Equal:
                this->dispatchArithmetic<EqualOperator>(symbols);
                break;
            case CompareOperation::NotEqual:
                this->dispatchArithmetic<NotEqualOperator>(symbols);
                break;
            case CompareOperation::Greater:
                this->dispatchArithmetic<GreaterThanOperator>(symbols);
                break;
            case CompareOperation::Lower:
                this->dispatchArithmetic<LessThanOperator>(symbols);
                break;
            case CompareOperation::GreaterEqual:
                this->dispatchArithmetic<GreaterEqualOperator>(symbols);
                break;
            case CompareOperation::LowerEqual:
                this->dispatchArithmetic<LessEqualOperator>(symbols);
                break;
            default:
                break;
            }
        }
	};
}

#endif
