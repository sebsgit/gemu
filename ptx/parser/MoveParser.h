#ifndef PTXMOVEPARSERH
#define PTXMOVEPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/memory/Move.h"

namespace ptx {
	namespace parser {
		class MoveParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				if (tokens.peek() == "mov") {
					Type type = Type::Unknown;
					size_t size = 0;
					tokens.removeFirst();
					if (Utils::parseTypeAndSize(tokens, &type, &size)) {
						MemoryInstructionOperand op1, op2;
						if (Utils::parseOperand(tokens, &op1)){
							if (tokens.peek()==",") {
								tokens.removeFirst();
								if (Utils::parseOperand(tokens, &op2)) {
									MemoryInstruction instr;
									instr.setType(type);
									instr.setSize(size);
									instr.setAllocSpace(AllocSpace::Register);
									instr.addOperand(op1);
									instr.addOperand(op2);
									result.add(std::make_shared<ptx::Move>(std::move(instr)));
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
