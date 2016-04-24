#ifndef PTXSUBINTPARSERH
#define PTXSUBINTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/int/Sub.h"

namespace ptx {
    namespace parser {
        class SubParser : public AbstractParser{
        protected:
            bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				if (this->standardParse<Sub>("sub", 3, tokens, result))
					return true;
                if (tokens.poll("sub")) {
                    if (tokens.poll(".sat")) {
                        //TODO
					}
                }
                return false;
            }
        };
    }
}

#endif
