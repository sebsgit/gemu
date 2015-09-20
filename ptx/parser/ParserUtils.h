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
						*size = 8;
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
				auto name = temp.peek();
				if (!isIdentifier(name))
					return false;
				temp.removeFirst();
				if (name == "%tid") {
					if (temp.peek() == ".x" || temp.peek() == ".y" || temp.peek() == ".z") {
						name += temp.peek();
						temp.removeFirst();
					} else {
						return false;
					}
				}
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
			static bool parseCompareOperator(TokenList& tokens, CompareOperation * result) {
				*result = CompareOperation::NotValidCompareOperation;
				const auto token = tokens.peek();
				if (token == ".eq") *result = CompareOperation::Equal;
				else if (token == ".ne") *result = CompareOperation::NotEqual;
				else if (token == ".gt") *result = CompareOperation::Greater;
				else if (token == ".lt") *result = CompareOperation::Lower;
				else if (token == ".ge") *result = CompareOperation::GreaterEqual;
				else if (token == ".lo") *result = CompareOperation::LowerEqual;
				if( *result != CompareOperation::NotValidCompareOperation ){
					tokens.removeFirst();
					return true;
				}
				return false;
			}
			static bool parseBooleanOperation(TokenList& tokens, BooleanOperation * result) {
				*result = BooleanOperation::NotValidBooleanOperation;
				const auto token = tokens.peek();
				if (token == ".and") *result = BooleanOperation::And;
				else if (token == ".or") *result = BooleanOperation::Or;
				else if (token == ".xor") *result = BooleanOperation::Xor;
				if (*result != BooleanOperation::NotValidBooleanOperation){
					tokens.removeFirst();
					return true;
				}
				return false;
			}
			static bool isIdentifier(const TokenList::token_t& token) {
				if (token.empty()==false) {
					size_t start = 0;
					if (token[0] == '%')
						start = 1;
					for (size_t i=start ; i<token.length() ; ++i) {
						const char c = token[i];
						if (c >= 'a' && c<='z') continue;
						if (c >= 'A' && c<='Z') continue;
						if (c >= '0' && c<='9') continue;
						if (c == '$' || c=='_') continue;
						return false;
					}
					return true;
				}
				return false;
			}
		};
	}
}

#endif
