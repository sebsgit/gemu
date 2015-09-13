#ifndef PTXPARSERUTILITIESH
#define PTXPARSERUTILITIESH

#include "semantics/globals.h"
#include "semantics/Instruction.h"
#include "Tokenizer.h"

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
			static bool parseVectorType(const std::string& token, VectorType * result) {
				*result = VectorType::VecNone;
				if (token == ".vec2")
					*result = VectorType::Vec2;
				else if (token == ".vec4")
					*result = VectorType::Vec4;
				return *result != VectorType::VecNone;
			}
			static bool parseCacheOperation(const std::string& token, CacheOperation * result) {
				*result = CacheOperation::CacheUndefined;
				if (token == ".ca") *result = CacheOperation::CacheAllLevels;
				else if (token == ".cg") *result = CacheOperation::CacheGlobal;
				else if (token == ".cs") *result = CacheOperation::CacheStreaming;
				else if (token == ".lu") *result = CacheOperation::CacheLastUse;
				else if (token == ".cv") *result = CacheOperation::CacheVolatile;
				else if (token == ".wb") *result = CacheOperation::CacheWriteBack;
				else if (token == ".wt") *result = CacheOperation::CacheWriteThrough;
				return *result != CacheOperation::CacheUndefined;
			}
			static bool parseOperand(TokenList& tokens, MemoryInstructionOperand& result) {
				TokenList temp = tokens;
				bool isAddressed = false;
				if (temp.peek() == "[") {
					isAddressed = true;
					temp.removeFirst();
				}
				// if (temp.peek is valid identifier)
				std::string name = temp.peek();
				temp.removeFirst();
				if (isAddressed && temp.peek() == "]"){
					temp.removeFirst();
				}
				tokens = temp;
				result = MemoryInstructionOperand(name, isAddressed, 0);
				return true;
			}
		};
	}
}

#endif
