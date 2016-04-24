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
						if (this->standardParse<AtomicAdd>(".add", 3, tokens, result))
							return true;
					}
				}
				return false;
			}
		};
	}
}

#endif
