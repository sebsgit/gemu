#ifndef PTXPARSERH
#define PTXPARSERH

#include <iostream>
#include <cassert>
#include <algorithm>
#include <sstream>
#include "Tokenizer.h"
#include "semantics/Semantics.h"
#include "parser/AbstractParser.h"
#include "parser/DirectiveParser.h"
#include "parser/FunctionParser.h"

namespace ptx {
	class Parser : public parser::AbstractParser {
		typedef TokenList::token_t token_t;
	public:
		~Parser(){}
		ParserResult parseModule(const std::string& source){
			ParserResult result;
			auto tokens = Tokenizer().tokenize(source);
			this->parse(tokens, result);
			return result;
		}
	private:
		bool parse(TokenList& tokens, ParserResult& result) const {
			parser::DirectiveParser directive;
			while (directive.parse(tokens, result));
			parser::FunctionParser functions;
			while (functions.parse(tokens, result));
			return true;
		}
	};
}

#endif
