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
		void addParameter(const ptx::Variable& var){
			this->_parameters._data.push_back(var);
		}
	private:
		std::vector<ptx::InstructionPtr> _instructions;
		FunctionParameters _parameters;
	};
}

#endif
