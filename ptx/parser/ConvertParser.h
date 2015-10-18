#ifndef PTXCONVERTPARSERH
#define PTXCONVERTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/memory/Convert.h"

namespace ptx {
	namespace parser {
		class ConvertParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
                if (tokens.poll("cvta")) {
                    if (tokens.poll(".to")) {
						//TODO
					}
					AllocSpace space = AllocSpace::Undefined;
					Type type = Type::Unknown;
					size_t size = 0;
					if (Utils::parseAllocSpace(tokens, &space) && Utils::parseTypeAndSize(tokens, &type, &size)) {
						MemoryInstructionOperands operands;
						if (Utils::parseOperands(tokens, 2, &operands)){
							MemoryInstruction instr;
							instr.setAllocSpace(space);
							instr.setSize(size);
							instr.setType(type);
							instr.setOperands(operands);
							result.add(std::make_shared<ptx::Convert>(std::move(instr)));
							return true;
						}
					}
                } else if (tokens.poll("cvt")) {
					//TODO implement rounding modes
                    if (tokens.poll(".rn")) {
						Type type;
						size_t size;
						if (Utils::parseTypeAndSize(tokens, &type, &size) && Utils::parseTypeAndSize(tokens, &type, &size)) {
							MemoryInstructionOperands operands;
							if (Utils::parseOperands(tokens, 2, &operands)){
								MemoryInstruction instr;
								instr.setOperands(operands);
								result.add(std::make_shared<ptx::ConvertValue>(std::move(instr)));
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
