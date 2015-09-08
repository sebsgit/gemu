#ifndef PTXFUNCTIONPARSERH
#define PTXFUNCTIONPARSERH

#include "parser/AbstractParser.h"

namespace ptx {
	namespace parser {
		class FunctionParser : public AbstractParser {
		public:
			bool parse(TokenList& tokens, ParserResult& result) const {
				return true;
			}
		};
	}
}

#endif
