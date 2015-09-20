#ifndef PTXBRANCHPARSERH
#define PTXBRANCHPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/control/Branch.h"

namespace ptx {
	namespace parser {
		class BranchParser : public AbstractParser{
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				if (tokens.peek() == "bra") {
					bool isDivergent = true;
					tokens.removeFirst();
					if (tokens.peek() == ".uni") {
						tokens.removeFirst();
						isDivergent = false;
					}
					const std::string label = tokens.peek();
					tokens.removeFirst();
					result.add(std::make_shared<ptx::Branch>(label, isDivergent));
					return true;
				}
				return false;
			}
		};
	}
}

#endif
