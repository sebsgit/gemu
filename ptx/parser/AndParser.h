#ifndef PTXBITANDINTPARSERH
#define PTXBITANDINTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/int/And.h"

namespace ptx {
	namespace parser {
		class AndParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				return this->standardParse<BitAnd>("and", 3, tokens, result);
			}
		};
	}
}

#endif
