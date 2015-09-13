#ifndef PTXRETEXITPARSERH
#define PTXRETEXITPARSERH

#include "parser/AbstractParser.h"
#include "semantics/instructions/control/Return.h"

namespace ptx {
	namespace parser {
		class ReturnExitParser : public AbstractParser{
		public:
			bool parse(TokenList& tokens, ParserResult& result) const override {
				if (tokens.peek() == "ret") {
					bool isDivergent = true;
					tokens.removeFirst();
					if (tokens.peek() == ".uni") {
						tokens.removeFirst();
						isDivergent = false;
					}
					result.add(std::make_shared<ptx::Return>(isDivergent));
					return true;
				}
				return false;
			}
		};
	}
}

#endif
