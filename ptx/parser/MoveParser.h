#ifndef PTXMOVEPARSERH
#define PTXMOVEPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/memory/Move.h"

namespace ptx {
	namespace parser {
		class MoveParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
                if (tokens.poll("mov")) {
					Type type = Type::Unknown;
					size_t size = 0;
					if (Utils::parseTypeAndSize(tokens, &type, &size)) {
						MemoryInstructionOperands operands;
						if (Utils::parseOperands(tokens, 2, &operands)){
							MemoryInstruction instr;
							instr.setType(type);
							instr.setSize(size);
							instr.setAllocSpace(AllocSpace::Register);
							instr.setOperands(operands);
							result.add(std::make_shared<ptx::Move>(std::move(instr)));
							return true;
						}
					}
				}
				return false;
			}
		};
	}
}

#endif
