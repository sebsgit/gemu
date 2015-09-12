#ifndef PTXFUNCTIONPARSERH
#define PTXFUNCTIONPARSERH

#include "parser/AbstractParser.h"
#include "parser/VariableParser.h"
#include "semantics/Function.h"
#include "semantics/instructions/FunctionDeclaration.h"

namespace ptx {
	namespace parser {
		class FunctionParser : public AbstractParser {
		public:
			bool parse(TokenList& tokens, ParserResult& result) const {
				if (tokens.empty())
					return false;
				bool toReturn = true;
				TokenList toRevert(tokens);
				Function function;
				function.setAllocSpace(AllocSpace::local);
				if (tokens.peek() == ".visible") {
					function.setAllocSpace(AllocSpace::shared);
					tokens.removeFirst();
				}
				if (tokens.peek() == ".entry") {
					function.setName(tokens.peek(1));
					tokens.removeFirst(2);
					TokenList varDecl = tokens.sublist("(", ")");
					if (varDecl.empty() == false) {
						ParserResult vars;
						if (VariableListParser().parse(varDecl, vars)) {
							tokens.removeUntilWith(")");
							for (size_t i=0 ; i<vars.count() ; ++i){
								function.addParameter(vars.fetch<ptx::VariableDeclaration>(i)->var());
							}
						} else {
							toReturn = false;
						}
					}
					if (toReturn) {
						TokenList body = tokens.sublist("{", "}");
						if (body.empty()==false) {

							tokens.removeUntilWith("}");
						} else {
							toReturn = false;
						}
					}
				}
				if (!toReturn) {
					tokens = toRevert;
				} else {
					result.add(std::make_shared<ptx::FunctionDeclaration>(function));
				}
				return toReturn;
			}
		};
	}
}

#endif
