#ifndef PTXFUNCTIONPARSERH
#define PTXFUNCTIONPARSERH

#include "parser/AbstractParser.h"
#include "parser/VariableParser.h"
#include "semantics/Function.h"

namespace ptx {
	namespace parser {
		class FunctionParser : public AbstractParser {
		public:
			bool parse(TokenList& tokens, ParserResult& result) const {
				bool toReturn = false;
				TokenList toRevert(tokens);
				AllocSpace space = AllocSpace::global;
				if (tokens.peek() == ".visible") {
					space = AllocSpace::shared;
					tokens.removeFirst();
				}
				if (tokens.peek() == ".entry") {
					Function function(tokens.peek(1));
					tokens.removeFirst(2);
					TokenList varDecl = tokens.sublist("(", ")");
					std::vector<ptx::Variable> parameters;
					if (varDecl.empty() == false) {
						if (VariableParser().parse(varDecl, parameters)) {
							function.setParameters(parameters);
							tokens.removeUntil(")");
						} else {
							toReturn = false;
						}
					}
					if (toReturn) {
						TokenList body = tokens.sublist("{", "}");
						if (body.empty()==false) {

						} else {
							toReturn = false;
						}
					}
				}
				if (!toReturn) {
					tokens = toRevert;
				}
				return toReturn;
			}
		};
	}
}

#endif
