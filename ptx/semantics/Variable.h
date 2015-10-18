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
	public:
		Variable()
			:Variable(AllocSpace::Undefined, Type::Unknown, 0, std::string())
		{}
		Variable (AllocSpace space, Type type, size_t size, const std::string& name)
			:_space(space)
			,_type(type)
			,_size(size)
			,_name(name)
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
        ptx::Variable renamed(const std::string& name) const {
            return ptx::Variable(this->space(), this->type(), this->size(), name);
        }
	};
}

#endif
