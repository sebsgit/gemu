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
		bool empty() const {
			return this->_instructions.empty();
		}
		size_t count() const {
			return this->_instructions.size();
		}
		template <typename T=ptx::Instruction>
		std::shared_ptr<T> fetch(size_t i = 0) const {
			return std::dynamic_pointer_cast<T>(this->_instructions.at(i));
		}
		void add(const ptx::InstructionPtr& toAdd) {
			this->_instructions.push_back(toAdd);
		}
		void add(const std::vector<ptx::InstructionPtr>& toAdd) {
			for (const auto& i : toAdd)
				this->add(i);
		}
		void add(const ParserResult& other) {
			for (const auto& i : other._instructions)
				this->add(i);
		}
	private:
		std::vector<ptx::InstructionPtr> _instructions;
	};
}

#endif
