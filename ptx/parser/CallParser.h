#ifndef PTXCALLPARSERH
#define PTXCALLPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/control/Call.h"

namespace ptx {
    namespace parser {
        class CallParser : public AbstractParser{
        protected:
            bool parseTokens(TokenList& tokens, ParserResult& result) const override {
                if (tokens.peek() == "call") {
                    bool isDivergent = true;
                    std::string callResult;
                    std::string callTarget;
                    std::vector<std::string> callParameters;
                    tokens.removeFirst();
                    if (tokens.peek() == ".uni") {
                        tokens.removeFirst();
                        isDivergent = false;
                    }
                    if (tokens.peek() == "(") {
                        tokens.removeFirst();
                        if (Utils::isIdentifier(tokens.peek())) {
                            callResult = tokens.peek();
                            tokens.removeFirst();
                            if (tokens.peek() == ")" && tokens.peek(1) == ",") {
                                tokens.removeFirst(2);
                            }
                        }
                    }
                    if (Utils::isIdentifier(tokens.peek())) {
                        callTarget = tokens.peek();
                        tokens.removeFirst();
                        if (tokens.peek() == ",") {
                            tokens.removeFirst();
                            if (tokens.peek() == "(") {
                                tokens.removeFirst();
                                while (tokens.peek() != ")"){
                                    if (Utils::isIdentifier(tokens.peek())){
                                        callParameters.push_back(tokens.peek());
                                        tokens.removeFirst();
                                        if (tokens.peek() == ","){
                                            tokens.removeFirst();
                                        }
                                    }
                                }
                                tokens.removeFirst();
                            }
                        }
                    } else {
                        return false;
                    }
                    result.add(std::make_shared<ptx::Call>(callTarget, callResult, callParameters, isDivergent));
                    return true;
                }
                return false;
            }
        };
    }
}

#endif
