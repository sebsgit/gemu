#ifndef PTXMULINTPARSERH
#define PTXMULINTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/int/Mul.h"

namespace ptx {
	namespace parser {
		class MulParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
                if (tokens.poll("mul")) {
                    if (tokens.poll(".wide") || tokens.poll(".lo")) {
						//TODO
					}
					Type type;
					size_t size;
					if (Utils::parseTypeAndSize(tokens, &type, &size)) {
						MemoryInstructionOperands operands;
						if (Utils::parseOperands(tokens, 3, &operands)) {
							MemoryInstruction instr;
							instr.setType(type);
							instr.setSize(size);
							instr.setOperands(operands);
							result.add(std::make_shared<Mul>(std::move(instr)));
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
