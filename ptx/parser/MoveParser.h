#ifndef PTXMOVEPARSERH
#define PTXMOVEPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/memory/Move.h"

namespace ptx {
	namespace parser {
		class MoveParser : public AbstractParser{
		public:
			bool parse(TokenList& tokens, ParserResult& result) const override {
				return false;
			}
		};
	}
}

#endif
