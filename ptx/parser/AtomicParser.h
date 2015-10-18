#ifndef PTXATOMICINTPARSERH
#define PTXATOMICINTPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/int/Atomic.h"

namespace ptx {
	namespace parser {
		class AtomicParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				if (tokens.poll("atom")) {
					AllocSpace space;
					if (Utils::parseAllocSpace(tokens,&space) && (space==AllocSpace::Shared || space==AllocSpace::Global)){
						//TODO other atomics
						if (tokens.poll(".add")) {
							Type type;
							size_t size;
							if (Utils::parseTypeAndSize(tokens, &type, &size)) {
								MemoryInstructionOperands operands;
								if (Utils::parseOperands(tokens, 3, &operands)) {
									MemoryInstruction instr;
									instr.setType(type);
									instr.setSize(size);
									instr.setOperands(operands);
									result.add(std::make_shared<AtomicAdd>(std::move(instr)));
									return true;
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
