#ifndef PTXFUNCTIONPARSERH
#define PTXFUNCTIONPARSERH

#include "parser/AbstractParser.h"
#include "parser/VariableParser.h"

namespace ptx {
	namespace parser {
		class FunctionParser : public AbstractParser {
		public:
			bool parse(TokenList& tokens, ParserResult& result) const {
				bool toReturn = false;
				TokenList toRevert(tokens);
				if (tokens.peek() == ".visible") {
					
					tokens.removeFirst();
				}
				if (tokens.peek() == ".entry") {
					tokens.removeFirst(2);
					TokenList varDecl = tokens.sublist("(", ")");
					if (varDecl.empty() == false) {
						if (VariableParser().parse(varDecl, result)) {
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
