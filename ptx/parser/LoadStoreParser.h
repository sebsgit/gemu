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
				return false;
			}
		};
	}
}

#endif
