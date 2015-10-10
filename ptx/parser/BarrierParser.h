#ifndef PTXBARRIERPARSERH
#define PTXBARRIERPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/control/Barrier.h"

namespace ptx {
	namespace parser {
		class BarrierParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				if (tokens.peek() == "bar") {
					tokens.removeFirst();
					BarrierType type;
					if (Utils::parseBarrierType(tokens, &type)){
						MemoryInstructionOperands operands;
						MemoryInstruction instr;
						if (Utils::parseOperands(tokens, 1, &operands)){
							instr.setOperands(operands);
							const int id = atoi(operands[0].symbol().c_str());
							result.add(std::make_shared<ptx::Barrier>(id, type, std::move(instr)));
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
