#ifndef PTXSELPPARSERH
#define PTXSELPPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/compare/Selp.h"

namespace ptx {
    namespace parser {
        class SelpParser : public AbstractParser{
        protected:
            bool parseTokens(TokenList& tokens, ParserResult& result) const override {
                if (tokens.poll("selp")) {
                    Type type;
                    size_t size;
                    if (Utils::parseTypeAndSize(tokens, &type, &size)) {
                        MemoryInstructionOperands operands;
                        if (Utils::parseOperands(tokens, 4, &operands)) {
                            MemoryInstruction instr;
                            instr.setType(type);
                            instr.setSize(size);
                            instr.setOperands(operands);
                            result.add(std::make_shared<Selp>(std::move(instr)));
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
