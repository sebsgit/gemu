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

	class ParserResult {
	public:
		void add(const ptx::InstructionPtr& toAdd) {
			this->_instructions.push_back(toAdd);
		}
		void add(const std::vector<ptx::InstructionPtr>& toAdd) {
			for (const auto& i : toAdd)
				this->add(i);
		}
		ptx::InstructionPtr fetch(size_t i = 0) const {
			return this->_instructions.at(i);
		}
	private:
		std::vector<ptx::InstructionPtr> _instructions;
	};
}

#endif
