#ifndef PTXPARSERUTILITIESH
#define PTXPARSERUTILITIESH

#include "semantics/globals.h"

namespace ptx {
	namespace parser {
		class Utils {
		public:
			static AllocSpace parseAllocSpace(const std::string& str) {
				if (str == ".reg")
					return AllocSpace::Register;
				else if (str == ".sreg")
					return AllocSpace::SpecialRegister;
				else if (str == ".const")
					return AllocSpace::Constant;
				else if (str == ".global")
					return AllocSpace::Global;
				else if (str == ".local")
					return AllocSpace::Local;
				else if (str == ".param")
					return AllocSpace::Parameter;
				else if (str == ".shared")
					return AllocSpace::Shared;
				return AllocSpace::Undefined;
			}
			static bool parseAllocSpace(const std::string& str, AllocSpace * space) {
				*space = parseAllocSpace(str);
				return *space != AllocSpace::Undefined;
			}
			static bool parseTypeAndSize(const std::string& str, Type * type, size_t * size) {
				*size = 0;
				*type = Type::Unknown;
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
				return size!=0;
			}
		};
	}
}

#endif
