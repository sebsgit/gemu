#ifndef PTXFUNCTIONH
#define PTXFUNCTIONH

#include "semantics/Instruction.h"
#include "semantics/Variable.h"

namespace ptx {

	class FunctionParameters {
	public:
		ptx::Variable variable(size_t i) const{
			return this->_data.at(i);
		}
		const ptx::Variable operator[](size_t i) const {
			return this->_data.at(i);
		}
		ptx::Variable variable(const std::string& name) const{
			for (const auto& v : _data)
				if (v.name()==name)
					return v;
			return ptx::Variable();
		}
		size_t size() const {
			return this->_data.size();
		}
	private:
		std::vector<ptx::Variable> _data;
		friend class Function;
	};

	class Function : public InstructionList {
	public:
		Function(const std::string& name = std::string())
		:_name(name)
		{}
		std::string name() const { return this->_name; }
		AllocSpace space() const { return this->_space; }
		FunctionParameters parameters() const { return this->_parameters; }
        Variable returnVariable() const { return this->_returnVariable; }
		void addParameter(const ptx::Variable& var){
			this->_parameters._data.push_back(var);
		}
		void setParameters(const std::vector<ptx::Variable>& vars) {
			this->_parameters._data = vars;
		}
		void setAllocSpace(ptx::AllocSpace space){
			this->_space = space;
		}
		void setName(const std::string& name) {
			this->_name = name;
		}
        void setReturnVariable(const ptx::Variable& var) {
            this->_returnVariable = var;
        }

		bool isNull() const {
			return this->_name.empty();
		}
	private:
		std::string _name;
		FunctionParameters _parameters;
        Variable _returnVariable;
		AllocSpace _space = Undefined;
	};
}

#endif
