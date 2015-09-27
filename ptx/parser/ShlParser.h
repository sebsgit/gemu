#ifndef PTXSHLINTPARSERH
#define PTXSHLINTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/int/Shl.h"

namespace ptx {
	namespace parser {
		class ShlParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				if (tokens.peek() == "shl") {
					tokens.removeFirst();
					Type type;
					size_t size;
					if (Utils::parseTypeAndSize(tokens, &type, &size)) {
						MemoryInstructionOperands operands;
						if (Utils::parseOperands(tokens, 3, &operands)) {
							MemoryInstruction instr;
							instr.setType(type);
							instr.setSize(size);
							instr.setOperands(operands);
							result.add(std::make_shared<Shl>(std::move(instr)));
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
