#ifndef SETPPPARSERPTXHH
#define SETPPPARSERPTXHH

#include "parser/AbstractParser.h"
#include "semantics/instructions/memory/Load.h"

namespace ptx {
	namespace parser {
		class SetpParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
                if (tokens.poll("setp")) {
					CompareOperation cmpOp;
					if (Utils::parseCompareOperator(tokens, &cmpOp)) {
						BooleanOperation boolOp = BooleanOperation::NotValidBooleanOperation;
						Utils::parseBooleanOperation(tokens, &boolOp);
						Type type;
						size_t size;
						if (Utils::parseTypeAndSize(tokens, &type, &size)) {
							MemoryInstructionOperands operands;
							if (Utils::parseOperands(tokens, 3, &operands)) {
								CompareInstruction instr;
								instr.setOperands(operands);
								instr.setType(type);
								instr.setSize(size);
								instr.setCompare(cmpOp);
								instr.setBoolean(boolOp);
								result.add(std::make_shared<ptx::Setp>(std::move(instr)));
								return true;
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
