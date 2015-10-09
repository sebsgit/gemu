#ifndef PTXMADINTPARSERH
#define PTXMADINTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/int/Mad.h"

namespace ptx {
	namespace parser {
		class MadParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				if (tokens.peek() == "mad") {
					tokens.removeFirst();
					Type type;
					size_t size;
					if (tokens.peek() == ".wide" || tokens.peek() == ".lo" || tokens.peek() == ".hi") {
						//TODO
						tokens.removeFirst();
					}
					if (Utils::parseTypeAndSize(tokens, &type, &size)) {
						MemoryInstructionOperands operands;
						if (Utils::parseOperands(tokens, 4, &operands)) {
							MemoryInstruction instr;
							instr.setType(type);
							instr.setSize(size);
							instr.setOperands(operands);
							result.add(std::make_shared<Mad>(std::move(instr)));
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
