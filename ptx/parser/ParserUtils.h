#ifndef PTXPARSERUTILITIESH
#define PTXPARSERUTILITIESH

#include "semantics/globals.h"
#include "semantics/Instruction.h"
#include "semantics/instructions/memory/MemoryInstruction.h"
#include "Tokenizer.h"

namespace ptx {
	namespace parser {
		class Utils {
		public:
			static AllocSpace parseAllocSpace(const TokenList::token_t& str) {
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
			static bool parseAllocSpace(TokenList& tokens, AllocSpace * result) {
				*result = parseAllocSpace(tokens.peek());
				if (*result != AllocSpace::Undefined) {
					tokens.removeFirst();
					return true;
				}
				return false;
			}
			static bool parseTypeAndSize(TokenList& tokens, Type * type, size_t * size) {
				const auto str = tokens.peek();
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
						const TokenList::token_t sizeStr = str.substr(2,str.size()-2);
						*size = atoi(sizeStr.c_str());
						if (*size != 8 && *size!=16 && *size!=32 && *size!=64)
							*size = 0;
						else
							*size /= 8;
					}
				}
				if(size!=0) {
					tokens.removeFirst();
					return true;
				}
				return false;
			}
			static bool parseVectorType(TokenList& tokens, VectorType * result) {
				*result = VectorType::VecNone;
				if (tokens.peek() == ".vec2")
					*result = VectorType::Vec2;
				else if (tokens.peek() == ".vec4")
					*result = VectorType::Vec4;
				if(*result != VectorType::VecNone){
					tokens.removeFirst();
					return true;
				}
				return false;
			}
			static bool parseCacheOperation(TokenList& tokens, CacheOperation * result) {
				*result = CacheOperation::CacheUndefined;
				const auto token = tokens.peek();
				if (token == ".ca") *result = CacheOperation::CacheAllLevels;
				else if (token == ".cg") *result = CacheOperation::CacheGlobal;
				else if (token == ".cs") *result = CacheOperation::CacheStreaming;
				else if (token == ".lu") *result = CacheOperation::CacheLastUse;
				else if (token == ".cv") *result = CacheOperation::CacheVolatile;
				else if (token == ".wb") *result = CacheOperation::CacheWriteBack;
				else if (token == ".wt") *result = CacheOperation::CacheWriteThrough;
				if(*result != CacheOperation::CacheUndefined){
					tokens.removeFirst();
					return true;
				}
				return false;
			}
			static bool parseOperand(TokenList& tokens, MemoryInstructionOperand * result) {
				if (tokens.empty())
					return false;
				TokenList temp = tokens;
				bool isAddressed = false;
				if (temp.peek() == "[") {
					isAddressed = true;
					temp.removeFirst();
				}
				// if (temp.peek is valid identifier)
				auto name = temp.peek();
				temp.removeFirst();
				if (isAddressed && temp.peek() == "]"){
					temp.removeFirst();
				}
				tokens = temp;
				*result = MemoryInstructionOperand(name, isAddressed, 0);
				return true;
			}
			static bool parseOperands(TokenList& tokens, size_t requiredCount, MemoryInstructionOperands * result) {
				MemoryInstructionOperand op;
				MemoryInstructionOperands tmpResult;
				bool parsedOk = false;
				while (Utils::parseOperand(tokens, &op)) {
					tmpResult.add(op);
					if (tmpResult.count() == requiredCount) {
						parsedOk = true;
						break;
					}
					if (tokens.peek() != ",")
						break;
					tokens.removeFirst();
				}
				if (parsedOk)
					*result = tmpResult;
				return parsedOk;
			}
		};
	}
}

#endif
