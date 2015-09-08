#ifndef PTXPARSERH
#define PTXPARSERH

#include <iostream>
#include <cassert>
#include <algorithm>
#include <sstream>
#include "Tokenizer.h"
#include "semantics/Semantics.h"

namespace ptx {
	class ParserException : public std::exception {
	private:
		std::string _what;
	public:
		ParserException(const std::string& description)
			:_what(description)
		{}
		const char * what() const noexcept(true) override {
			return this->_what.c_str();
		}
	};
	class ParserResult {

	};
	class Parser {
	public:
		ParserResult parseModule(const std::string& source){
			ParserResult result;
			auto tokens = Tokenizer().tokenize(source);
			this->parseModule(tokens);
			return result;
		}
	private:
		void parseModule(TokenList& tokens) {
			if (tokens.peek() == ".version") {

			} else {
				throw ParserException(".version required!");
			}
		}
	};
}

#endif
