#ifndef PTXINSTRPARSERH
#define PTXINSTRPARSERH

#include "semantics/Instruction.h"
#include "parser/VariableParser.h"

namespace ptx {
	namespace parser {
		class InstructionParser : public AbstractParser {
		public:
			bool parse(TokenList& tokens, ParserResult& result) const override {
				tokens.clear();
				return true;
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
