#ifndef PTXINSTRPARSERH
#define PTXINSTRPARSERH

#include "semantics/Instruction.h"
#include "parser/AbstractParser.h"

namespace ptx {
	namespace parser {
		class InstructionParser : public AbstractParser {
		private:
			std::vector<std::shared_ptr<AbstractParser>> parsers;
		public:
            InstructionParser();
		protected:
            bool parseTokens(TokenList& tokens, ParserResult& result) const override;
		};
		class InstructionBlockParser : public SplittingParser<InstructionParser> {
		public:
            InstructionBlockParser();
		};
        class InstructionBracketParser : public AbstractParser {
        private:
            bool parseTokens(TokenList& tokens, ParserResult& result) const override {
                if (tokens.peek() == "{") {
                    auto toParse = tokens.bracketBody();
                    if (InstructionBlockParser().parse(toParse, result)) {
                        tokens.removeBracketBody();
                        return true;
                    }
                }
                return false;
            }
        };
	}
}

#endif
