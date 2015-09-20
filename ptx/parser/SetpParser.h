#ifndef SETPPPARSERPTXHH
#define SETPPPARSERPTXHH

#include "parser/AbstractParser.h"
#include "semantics/instructions/memory/Load.h"

namespace ptx {
	namespace parser {
		class SetpParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				if (tokens.peek() == "setp") {
					tokens.removeFirst();
					CompareOperation cmpOp;
					if (Utils::parseCompareOperator(tokens, &cmpOp)) {
						BooleanOperation boolOp = BooleanOperation::NotValidBooleanOperation;
						if (Utils::parseBooleanOperation(tokens, &boolOp)) {

						} else {
							Type type;
							size_t size;
							if (Utils::parseTypeAndSize(tokens, &type, &size)) {
								MemoryInstructionOperands operands;
								if (Utils::parseOperands(tokens, 3, &operands)) {
									// instr.setOperands(operands);
									// result.add(std::make_shared<ptx::Load>(std::move(instr)));
									return true;
								}
							}
						}
					}
				}
				return false;
			}
		};
	}
}
#endif
