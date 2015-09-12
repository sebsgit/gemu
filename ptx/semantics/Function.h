#ifndef PTXFUNCTIONH
#define PTXFUNCTIONH

#include "semantics/Instruction.h"
#include "semantics/Variable.h"

namespace ptx {

	class FunctionParameters {
	public:
		ptx::Variable variable(const std::string& name) const{
			for (const auto& v : _data)
				if (v.name()==name)
					return v;
			return ptx::Variable();
		}
	private:
		std::vector<ptx::Variable> _data;
		friend class Function;
	};

	class Function {
	public:
		Function(const std::string& name = std::string())
		:_name(name)
		{}
		std::string name() const { return this->_name; }
		void addParameter(const ptx::Variable& var){
			this->_parameters._data.push_back(var);
		}
		void setParameters(const std::vector<ptx::Variable>& vars) {
			this->_parameters._data = vars;
		}
		void addInstruction(const ptx::InstructionPtr& i) {
			this->_instructions.push_back(i);
		}
		void addInstructions(const std::vector<ptx::InstructionPtr>& toAdd) {
			for (const auto& i: toAdd)
				this->_instructions.push_back(i);
		}
		void setAllocSpace(ptx::AllocSpace space){
			this->_space = space;
		}
		void setName(const std::string& name) {
			this->_name = name;
		}
	private:
		std::string _name;
		std::vector<ptx::InstructionPtr> _instructions;
		FunctionParameters _parameters;
		AllocSpace _space = undefined;
	};
}

#endif
