#ifndef PTXINSTRPARSERH
#define PTXINSTRPARSERH

#include "semantics/Instruction.h"
#include "parser/VariableParser.h"
#include "parser/StoreParser.h"
#include "parser/LoadParser.h"
#include "parser/MoveParser.h"
#include "parser/ConvertParser.h"
#include "parser/ReturnExitParser.h"
#include "parser/BranchParser.h"
#include "parser/SetpParser.h"

namespace ptx {
	namespace parser {
		class InstructionParser : public AbstractParser {
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				bool parsedOk = false;
				std::vector<std::shared_ptr<AbstractParser>> parsers;
				parsers.push_back(std::make_shared<VariableParser>());
				parsers.push_back(std::make_shared<StoreParser>());
				parsers.push_back(std::make_shared<LoadParser>());
				parsers.push_back(std::make_shared<MoveParser>());
				parsers.push_back(std::make_shared<ConvertParser>());
				parsers.push_back(std::make_shared<ReturnExitParser>());
				parsers.push_back(std::make_shared<BranchParser>());
				parsers.push_back(std::make_shared<SetpParser>());
				std::string label;
				std::string predicate;
				Utils::parseLabel(tokens, &label);
				Utils::parsePredicate(tokens, &predicate);
				for (const auto& p : parsers) {
					if (p->parse(tokens, result)) {
						if (label.empty()==false)
							result.labelLast(label);
						if (predicate.empty()==false)
							result.predicateLast(predicate);
						parsedOk = true;
						break;
					}
				}
				return parsedOk;
			}
		};
		class InstructionBlockParser : public SplittingParser<InstructionParser> {
		public:
			InstructionBlockParser()
				:SplittingParser<InstructionParser>(token_t(";"))
			{}
		};
	}
}

#endif
