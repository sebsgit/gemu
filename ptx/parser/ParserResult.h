#ifndef PTXPARSERRESULTH
#define PTXPARSERRESULTH

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

	class ParserResult : public InstructionList {
	};
}

#endif
