#ifndef PTXSUBINTPARSERH
#define PTXSUBINTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/int/Sub.h"

namespace ptx {
    namespace parser {
        class SubParser : public AbstractParser{
        protected:
            bool parseTokens(TokenList& tokens, ParserResult& result) const override {
                if (tokens.peek() == "sub") {
                    tokens.removeFirst();
                    if (tokens.peek() == ".sat") {
                        //TODO
                    } else {
                        Type type;
                        size_t size;
                        if (Utils::parseTypeAndSize(tokens, &type, &size)) {
                            MemoryInstructionOperands operands;
                            if (Utils::parseOperands(tokens, 3, &operands)) {
                                MemoryInstruction instr;
                                instr.setType(type);
                                instr.setSize(size);
                                instr.setOperands(operands);
                                result.add(std::make_shared<Sub>(std::move(instr)));
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
