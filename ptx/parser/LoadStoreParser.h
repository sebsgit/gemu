#ifndef PTXLOADSTOREPARSERH
#define PTXLOADSTOREPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/memory/Load.h"
#include "semantics/instructions/memory/Store.h"

namespace ptx {
	namespace parser {
		class LoadStoreParser : public AbstractParser{
		public:
			bool parse(TokenList& tokens, ParserResult& result) const override {
				if (tokens.peek() == "ld" || tokens.peek() == "st") {
					bool isLoad = tokens.peek()=="ld";
					TokenList temp = tokens;
					temp.removeFirst();
					CacheOperation cacheOp = CacheOperation::CacheAllLevels;
					AllocSpace space = AllocSpace::Undefined;
					Type type = Type::Unknown;
					VectorType vecType = VectorType::VecNone;
					size_t size=0;
					MemoryInstruction instr;
					if  (temp.peek(1) == ".volatile"){
						temp.removeFirst();
						instr.setVolatile();
					}
					if (Utils::parseAllocSpace(temp.peek(), &space)){
						temp.removeFirst();
						if (!instr.isVolatile()) {
							if (Utils::parseCacheOperation(temp.peek(), &cacheOp)) {
								instr.setCacheMode(cacheOp);
								temp.removeFirst();
							}
						}
						if (Utils::parseVectorType(temp.peek(), &vecType)){
							temp.removeFirst();
						}
						if (Utils::parseTypeAndSize(temp.peek(), &type, &size)) {
							temp.removeFirst();
							instr.setType(type);
							instr.setSize(size);
							instr.setAllocSpace(space);
							instr.setVectorized(vecType);
							MemoryInstructionOperand op1, op2;
							if (Utils::parseOperand(temp, op1)) {
								if (temp.peek()==","){
									temp.removeFirst();
									if (Utils::parseOperand(temp, op2)) {
										tokens = temp;
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
