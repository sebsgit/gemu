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
		static AllocSpace parseAllocSpace(const std::string& str) {
			if (str == ".reg")
				return AllocSpace::reg;
			else if (str == ".sreg")
				return AllocSpace::sreg;
			else if (str == ".const")
				return AllocSpace::const_;
			else if (str == ".global")
				return AllocSpace::global;
			else if (str == ".local")
				return AllocSpace::local;
			else if (str == ".param")
				return AllocSpace::param;
			else if (str == ".shared")
				return AllocSpace::shared;
			return AllocSpace::undefined;
		}
		static void parseTypeAndSize(const std::string& str, Type * type, size_t * size) {
			*size = 0;
			*type = Type::Undefined;
			if (str.size() > 2) {
				if (str == ".pred") {
					*type = Type::Predicate;
					*size = 1;
				} else {
					const char c = str[1];
					if (c == 'u')
						*type = Type::Unsigned;
					else if (c == 's')
						*type = Type::Signed;
					else if (c == 'b')
						*type = Type::Bits;
					else if (c == 'f')
						*type = Type::Float;
					const std::string sizeStr = str.substr(2,str.size()-2);
					*size = atoi(sizeStr.c_str());
					if (*size != 8 && *size!=16 && *size!=32 && *size!=64)
						*size = 0;
				}
			}
		}
	public:
		Variable (AllocSpace space, Type type, size_t size, const std::string& name)
			:_space(space)
			,_type(type)
			,_size(size)
			,_name(name)
		{}
		Variable (const std::string& space, const std::string& type, const std::string& name)
			:_space(parseAllocSpace(space))
			,_name(name)
		{
			parseTypeAndSize(type, &_type, &_size);
		}
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
	};
}

#endif
