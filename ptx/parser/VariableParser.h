#ifndef PTXVARSPARSERH
#define PTXVARSPARSERH

#include "semantics/Variable.h"
#include "parser/AbstractParser.h"
#include "semantics/instructions/VariableDeclaration.h"

namespace ptx {
	namespace parser {
		class VariableParser : public AbstractParser {
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				AllocSpace space = AllocSpace::Undefined;
				if (Utils::parseAllocSpace(tokens, &space)) {
					ptx::Type type;
					size_t size = 0;
					if  (Utils::parseTypeAndSize(tokens, &type, &size)) {
						std::string name = tokens.peek();
						tokens.removeFirst();
						result.add(std::make_shared<ptx::VariableDeclaration>(ptx::Variable(space, type, size, name)));
						if (tokens.peek() == ";")
							tokens.removeFirst();
						return true;
					}
				}
				return false;
			}
		};

		class VariableListParser : public SplittingParser<VariableParser> {
		public:
			VariableListParser()
				:SplittingParser<VariableParser>(token_t(","))
			{
			}
		};
	}
}

#endif
