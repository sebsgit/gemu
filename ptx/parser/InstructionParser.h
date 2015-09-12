#ifndef PTXINSTRPARSERH
#define PTXINSTRPARSERH

#include "semantics/Instruction.h"
#include "parser/VariableParser.h"
#include "parser/LoadStoreParser.h"
#include "parser/MoveParser.h"
#include "parser/ConvertParser.h"

namespace ptx {
	namespace parser {
		class InstructionParser : public AbstractParser {
		public:
			bool parse(TokenList& tokens, ParserResult& result) const override {
				bool parsedOk = false;
				const TokenList toRevert(tokens);
				ParserResult partialResult;
				std::vector<std::shared_ptr<AbstractParser>> parsers;
				parsers.push_back(std::make_shared<VariableParser>());
				parsers.push_back(std::make_shared<LoadStoreParser>());
				parsers.push_back(std::make_shared<MoveParser>());
				parsers.push_back(std::make_shared<ConvertParser>());
				for (const auto& p : parsers) {
					if (p->parse(tokens, partialResult)) {
						parsedOk = true;
						result.add(partialResult);
						break;
					}
				}
				if (!parsedOk)
					tokens = toRevert;

				tokens.clear();
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
