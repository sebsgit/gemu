#ifndef PTXBITORINTPARSERH
#define PTXBITORINTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/int/Or.h"

namespace ptx {
    namespace parser {
        class OrParser : public AbstractParser{
        protected:
            bool parseTokens(TokenList& tokens, ParserResult& result) const override {
                if (tokens.poll("or")) {
                    Type type;
                    size_t size;
                    if (Utils::parseTypeAndSize(tokens, &type, &size)) {
                        MemoryInstructionOperands operands;
                        if (Utils::parseOperands(tokens, 3, &operands)) {
                            MemoryInstruction instr;
                            instr.setType(type);
                            instr.setSize(size);
                            instr.setOperands(operands);
                            result.add(std::make_shared<BitOr>(std::move(instr)));
                            return true;
                        }
                    }
                }
                return false;
            }
        };

        class XorParser : public AbstractParser{
        protected:
            bool parseTokens(TokenList& tokens, ParserResult& result) const override {
                if (tokens.poll("xor")) {
                    Type type;
                    size_t size;
                    if (Utils::parseTypeAndSize(tokens, &type, &size)) {
                        MemoryInstructionOperands operands;
                        if (Utils::parseOperands(tokens, 3, &operands)) {
                            MemoryInstruction instr;
                            instr.setType(type);
                            instr.setSize(size);
                            instr.setOperands(operands);
                            result.add(std::make_shared<BitXor>(std::move(instr)));
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
