#ifndef PTXADDINTPARSERH
#define PTXADDINTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/int/Add.h"

namespace ptx {
	namespace parser {
		class AddParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				if (this->standardParse<Add>("add", 3, tokens, result))
					return true;
				if (tokens.poll("add")) {
					if (tokens.peek() == ".sat") {
						//TODO
					}
				}
				return false;
			}
		};
	}
}

#endif
