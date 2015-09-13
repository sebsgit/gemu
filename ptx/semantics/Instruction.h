#ifndef PTXINSTRUCTIONH
#define PTXINSTRUCTIONH

#include <memory>
#include <sstream>
#include "semantics/globals.h"

namespace ptx {
	class Instruction {
	public:
		virtual ~Instruction() {};
		virtual std::string toString() const{
			return std::string("[not implemented]");
		}
	protected:
	};
	typedef std::shared_ptr<ptx::Instruction> InstructionPtr;

	class InstructionList {
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
		template <typename T=ptx::Instruction>
		std::shared_ptr<T> last() const {
			return this->fetch<T>(this->_instructions.size()-1);
		}
		void add(const ptx::InstructionPtr& toAdd) {
			this->_instructions.push_back(toAdd);
		}
		void add(const InstructionList& other) {
			for (const auto& i : other._instructions)
				this->add(i);
		}
		void addInto(InstructionList& other) const {
			other.add(*this);
		}
		std::string toString() const {
			std::stringstream ss;
			for (const auto& i : this->_instructions)
				ss << i << "\n";
			return ss.str();
		}
	private:
		std::vector<ptx::InstructionPtr> _instructions;
	};
}

#endif
