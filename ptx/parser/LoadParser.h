#ifndef LOADPARSERPTXHH
#define LOADPARSERPTXHH

#include "parser/AbstractParser.h"
#include "semantics/instructions/memory/Load.h"

namespace ptx {
	namespace parser {
		class LoadParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				if (tokens.peek() == "ld") {
					tokens.removeFirst();
					CacheOperation cacheOp = CacheOperation::CacheAllLevels;
					AllocSpace space = AllocSpace::Undefined;
					Type type = Type::Unknown;
					VectorType vecType = VectorType::VecNone;
					size_t size=0;
					MemoryInstruction instr;
					if  (tokens.peek(1) == ".volatile"){
						tokens.removeFirst();
						instr.setVolatile();
					}
					if (Utils::parseAllocSpace(tokens, &space)){
						if (!instr.isVolatile()) {
							if (Utils::parseCacheOperation(tokens, &cacheOp)) {
								instr.setCacheMode(cacheOp);
							}
						}
						Utils::parseVectorType(tokens, &vecType);
						if (Utils::parseTypeAndSize(tokens, &type, &size)) {
							instr.setType(type);
							instr.setSize(size);
							instr.setAllocSpace(space);
							instr.setVectorized(vecType);
							MemoryInstructionOperands operands;
							if (Utils::parseOperands(tokens, 2, &operands)) {
								instr.setOperands(operands);
								result.add(std::make_shared<ptx::Load>(std::move(instr)));
								return true;
							}
						}
					}
				}
				return false;
			}
		};
	}
}

#endif
