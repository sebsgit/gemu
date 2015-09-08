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
	public:
		const Module module() const { return this->_module; }
		Module& module() { return this->_module; }
	private:
		Module _module;
	};

	class Parser {
		typedef TokenList::token_t token_t;
	public:
		ParserResult parseModule(const std::string& source){
			ParserResult result;
			auto tokens = Tokenizer().tokenize(source);
			this->parseModule(tokens, result);
			return result;
		}
	private:
		void parseModule(TokenList& tokens, ParserResult& result) const {
			while (parseDirective(tokens, result));
		}
		bool parseDirective(TokenList& tokens, ParserResult& result) const {
			if (isDirective(tokens.peek())) {
				if (tokens.peek() == ".version") {
					result.module().setVersion(atof(tokens.peek(1).c_str()));
					tokens.removeFirst(2);
					return true;
				} else if (tokens.peek() == ".target") {
					result.module().setTargetArch(tokens.peek(1));
					tokens.removeFirst(2);
					return true;
				} else if (tokens.peek() == ".address_size") {
					result.module().setAddressSize(atoi(tokens.peek(1).c_str()));
					tokens.removeFirst(2);
					return true;
				}
			}
			return false;
		}
		bool isDirective(const token_t& token) const {
			return token.at(0) == '.';
		}
	};
}

#endif
