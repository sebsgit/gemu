#ifndef PTXSELPPARSERH
#define PTXSELPPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/compare/Selp.h"

namespace ptx {
    namespace parser {
        class SelpParser : public AbstractParser{
        protected:
            bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				return this->standardParse<Selp>("selp", 4, tokens, result);
            }
        };
    }
}

#endif
