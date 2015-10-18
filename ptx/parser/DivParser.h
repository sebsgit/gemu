#ifndef PTXMDIVINTPARSERH
#define PTXDIVINTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/int/Div.h"

namespace ptx {
    namespace parser {
        class DivParser : public AbstractParser{
        protected:
            bool parseTokens(TokenList& tokens, ParserResult& result) const override {
                if (tokens.poll("div")) {
                    Type type;
                    size_t size;
                    if (Utils::parseTypeAndSize(tokens, &type, &size)) {
                        MemoryInstructionOperands operands;
                        if (Utils::parseOperands(tokens, 3, &operands)) {
                            MemoryInstruction instr;
                            instr.setType(type);
                            instr.setSize(size);
                            instr.setOperands(operands);
                            result.add(std::make_shared<Div>(std::move(instr)));
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
