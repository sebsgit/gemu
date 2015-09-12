#ifndef PTXMODULEH
#define PTXMODULEH

#include <string>
#include "semantics/Function.h"

namespace ptx {
	class Module {
	public:
		void setVersion(const float ver) { this->_version = ver; }
		float version() const { return this->_version; }
		void setTargetArch(const std::string& arch){ this->_targetArch = arch; }
		void addFunction(const ptx::Function& f){
			this->_functions.push_back(f);
		}
		std::string targetArch() const {return this->_targetArch; }
		void setAddressSize(const size_t size){ this->_addressSize = size; }
		size_t addressSize() const {return this->_addressSize; }
	private:
		std::vector<ptx::Function> _functions;
		float _version = 0.0f;
		std::string _targetArch;
		size_t _addressSize = 32;
	};
}

#endif
