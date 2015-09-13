#ifndef PTXCONVERTPARSERH
#define PTXCONVERTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/memory/Convert.h"

namespace ptx {
	namespace parser {
		class ConvertParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				if (tokens.peek() == "cvta") {
					tokens.removeFirst();
					if (tokens.peek() == ".to") {
						//TODO
						tokens.removeFirst();
					}
					AllocSpace space = AllocSpace::Undefined;
					Type type = Type::Unknown;
					size_t size = 0;
					if (Utils::parseAllocSpace(tokens, &space) && Utils::parseTypeAndSize(tokens, &type, &size)) {
						MemoryInstructionOperand op1, op2;
						if (Utils::parseOperand(tokens, &op1)){
							if (tokens.peek() == ",") {
								tokens.removeFirst();
								if (Utils::parseOperand(tokens, &op2)) {
									MemoryInstruction instr;
									instr.setAllocSpace(space);
									instr.setSize(size);
									instr.setType(type);
									instr.addOperand(op1);
									instr.addOperand(op2);
									result.add(std::make_shared<ptx::Convert>(std::move(instr)));
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
