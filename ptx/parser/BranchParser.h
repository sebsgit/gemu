#ifndef PTXBRANCHPARSERH
#define PTXBRANCHPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/control/Branch.h"

namespace ptx {
	namespace parser {
		class BranchParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
                if (tokens.poll("bra")) {
                    const bool isDivergent = !tokens.poll(".uni");
                    const std::string label = tokens.takeFirst();
					result.add(std::make_shared<ptx::Branch>(label, isDivergent));
					return true;
				}
				return false;
			}
		};
	}
}

#endif
