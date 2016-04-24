#ifndef PTXREMINTPARSERH
#define PTXREMINTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/int/Rem.h"

namespace ptx {
	namespace parser {
        class RemParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				return this->standardParse<Rem>("rem", 3, tokens, result);
			}
		};
	}
}

#endif
