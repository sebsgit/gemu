#ifndef PTXCALLPARSERH
#define PTXCALLPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/control/Call.h"

namespace ptx {
    namespace parser {
        class CallParser : public AbstractParser{
        protected:
            bool parseTokens(TokenList& tokens, ParserResult& result) const override {
                if (tokens.poll("call")) {
                    bool isDivergent = true;
                    std::string callResult;
                    std::string callTarget;
                    std::vector<std::string> callParameters;
                    if (tokens.poll(".uni")) {
                        isDivergent = false;
                    }
                    if (tokens.poll("(")) {
                        if (Utils::isIdentifier(tokens.peek())) {
                            callResult = tokens.takeFirst();
                            if (tokens.peek() == ")" && tokens.peek(1) == ",") {
                                tokens.removeFirst(2);
                            }
                        }
                    }
                    if (Utils::isIdentifier(tokens.peek())) {
                        callTarget = tokens.takeFirst();
                        if (tokens.poll(",")) {
                            if (tokens.poll("(")) {
                                while (tokens.peek() != ")"){
                                    if (Utils::isIdentifier(tokens.peek())){
                                        callParameters.push_back(tokens.takeFirst());
                                        tokens.poll(",");
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
