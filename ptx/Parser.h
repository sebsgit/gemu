#ifndef PTXPARSERH
#define PTXPARSERH

#include "Tokenizer.h"
#include "parser/AbstractParser.h"

namespace ptx {
	class Parser : public parser::AbstractParser {
		typedef TokenList::token_t token_t;
	public:
		~Parser(){}
        ParserResult parseModule(const std::string& source);
	protected:
        bool parseTokens(TokenList& tokens, ParserResult& result) const;
	};
}

#endif
