#ifndef PTXMDIVINTPARSERH
#define PTXDIVINTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/int/Div.h"

namespace ptx {
    namespace parser {
        class DivParser : public AbstractParser{
        protected:
            bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				return this->standardParse<Div>("div", 3, tokens, result);
            }
        };
    }
}

#endif
