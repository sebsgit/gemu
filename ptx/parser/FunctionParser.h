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
                bool toReturn = false;
				Function function;
				function.setAllocSpace(AllocSpace::Local);
                if (tokens.poll(".visible")) {
					function.setAllocSpace(AllocSpace::Shared);
				}
                if (tokens.poll(".entry") || tokens.poll(".func")) {
                    toReturn = true;
                    if (tokens.poll("(")) {
                        ParserResult var;
                        if (VariableParser().parse(tokens,var) && tokens.poll(")")) {
                            if (var.fetch<ptx::VariableDeclaration>(0)) {
                                function.setReturnVariable(var.fetch<ptx::VariableDeclaration>(0)->var());
                            }
                        }
                    }
                    function.setName(tokens.takeFirst());
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
                        TokenList body = tokens.bracketBody();
						ParserResult bodyResult;
						if (body.empty()==false) {
							if (InstructionBlockParser().parse(body, bodyResult)) {
                                tokens.removeBracketBody();
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
