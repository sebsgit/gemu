#ifndef PTXFUNCTIONPARSERH
#define PTXFUNCTIONPARSERH

#include "parser/AbstractParser.h"
#include "parser/VariableParser.h"
#include "parser/InstructionParser.h"
#include "semantics/Function.h"
#include "semantics/instructions/FunctionDeclaration.h"

namespace ptx {
	namespace parser {
		class FunctionParser : public AbstractParser {
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const {
				if (tokens.empty())
					return false;
				bool toReturn = true;
				Function function;
				function.setAllocSpace(AllocSpace::Local);
				if (tokens.peek() == ".visible") {
					function.setAllocSpace(AllocSpace::Shared);
					tokens.removeFirst();
				}
                if (tokens.peek() == ".entry" || tokens.peek() == ".func") {
                    tokens.removeFirst();
                    if (tokens.peek() == "(") {
                        tokens.removeFirst();
                        ParserResult var;
                        if (VariableParser().parse(tokens,var) && tokens.peek() == ")") {
                            tokens.removeFirst();
                            if (var.fetch<ptx::VariableDeclaration>(0)) {
                                function.setReturnVariable(var.fetch<ptx::VariableDeclaration>(0)->var());
                            }
                        }
                    }
                    function.setName(tokens.peek(0));
                    tokens.removeFirst();
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
						ParserResult bodyResult;
						if (body.empty()==false) {
							if (InstructionBlockParser().parse(body, bodyResult)) {
								tokens.removeUntilWith("}");
								function.add(bodyResult);
							} else {
								toReturn = false;
							}
						} else {
							toReturn = false;
						}
					}
				}
				if (toReturn) {
					result.add(std::make_shared<ptx::FunctionDeclaration>(function));
				}
				return toReturn;
			}
		};
	}
}

#endif
