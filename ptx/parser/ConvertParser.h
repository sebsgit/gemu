#ifndef PTXCONVERTPARSERH
#define PTXCONVERTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/memory/Convert.h"

namespace ptx {
	namespace parser {
		class ConvertParser : public AbstractParser{
		public:
			bool parse(TokenList& tokens, ParserResult& result) const override {
				return false;
			}
		};
	}
}

#endif
