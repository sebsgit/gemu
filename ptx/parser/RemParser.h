#ifndef PTXREMINTPARSERH
#define PTXREMINTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/int/Rem.h"

namespace ptx {
	namespace parser {
        class RemParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
                if (tokens.poll("rem")) {
                    Type type;
                    size_t size;
                    if (Utils::parseTypeAndSize(tokens, &type, &size)) {
                        MemoryInstructionOperands operands;
                        if (Utils::parseOperands(tokens, 3, &operands)) {
                            MemoryInstruction instr;
                            instr.setType(type);
                            instr.setSize(size);
                            instr.setOperands(operands);
                            result.add(std::make_shared<Rem>(std::move(instr)));
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
