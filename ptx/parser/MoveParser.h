#ifndef PTXMOVEPARSERH
#define PTXMOVEPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/memory/Move.h"

namespace ptx {
	namespace parser {
		class MoveParser : public AbstractParser{
		public:
			bool parse(TokenList& tokens, ParserResult& result) const override {
				TokenList temp = tokens;
				if (temp.peek() == "mov") {
					Type type = Type::Unknown;
					size_t size = 0;
					temp.removeFirst();
					if (Utils::parseTypeAndSize(temp, &type, &size)) {
						MemoryInstructionOperand op1, op2;
						if (Utils::parseOperand(temp, &op1)){
							if (temp.peek()==",") {
								temp.removeFirst();
								if (Utils::parseOperand(temp, &op2)) {
									tokens = temp;
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
