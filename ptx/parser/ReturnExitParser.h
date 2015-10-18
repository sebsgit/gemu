#ifndef PTXRETEXITPARSERH
#define PTXRETEXITPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/control/Return.h"

namespace ptx {
	namespace parser {
		class ReturnExitParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
                if (tokens.poll("ret")) {
                    const bool isDivergent = !tokens.poll(".uni");
					result.add(std::make_shared<ptx::Return>(isDivergent));
					return true;
				}
				return false;
			}
		};
	}
}

#endif
