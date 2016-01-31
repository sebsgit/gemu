#ifndef PTXSEMANTICSVARIABLEH
#define PTXSEMANTICSVARIABLEH

#include "semantics/globals.h"

namespace ptx {
	class Variable {
	private:
		AllocSpace _space;
		Type _type;
		size_t _size;
		std::string _name;
        size_t _arraySize;
	public:
		Variable()
			:Variable(AllocSpace::Undefined, Type::Unknown, 0, std::string())
		{}
        Variable (AllocSpace space, Type type, size_t size, const std::string& name, size_t arraySize = 1)
			:_space(space)
			,_type(type)
			,_size(size)
			,_name(name)
            ,_arraySize(arraySize)
		{}
		AllocSpace space() const {
			return this->_space;
		}
		Type type() const {
			return this->_type;
		}
		size_t size() const {
			return this->_size;
		}
		std::string name() const {
			return this->_name;
		}
        size_t arraySize() const {
            return this->_arraySize;
        }
        ptx::Variable renamed(const std::string& name) const {
            return ptx::Variable(this->space(), this->type(), this->size(), name);
        }
	};
}

#endif
