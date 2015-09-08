#ifndef PTXABSTRACTPARSERH
#define PTXABSTRACTPARSERH

#include "parser/ParserResult.h"
#include "Tokenizer.h"

namespace ptx {
	namespace parser {
		class AbstractParser {
		protected:
			typedef TokenList::token_t token_t;
		public:
			virtual ~AbstractParser(){}
			virtual bool parse(TokenList& tokens, ParserResult& result) const = 0;
		};
	}
}

#endif
