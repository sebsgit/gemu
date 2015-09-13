#ifndef PTXLOADSTOREPARSERH
#define PTXLOADSTOREPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/memory/Load.h"
#include "semantics/instructions/memory/Store.h"

namespace ptx {
	namespace parser {
		class LoadStoreParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				if (tokens.peek() == "ld" || tokens.peek() == "st") {
					bool isLoad = tokens.peek()=="ld";
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
							MemoryInstructionOperand op1, op2;
							if (Utils::parseOperand(tokens, &op1)) {
								if (tokens.peek()==","){
									tokens.removeFirst();
									if (Utils::parseOperand(tokens, &op2)) {
										instr.addOperand(op1);
										instr.addOperand(op2);
										if (isLoad)
											result.add(std::make_shared<ptx::Load>(std::move(instr)));
										else
											result.add(std::make_shared<ptx::Store>(std::move(instr)));
										return true;
									}
								}
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
